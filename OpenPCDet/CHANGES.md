# 感知误差分布提取（Perception Error Extraction）

本文档记录在 OpenPCDet 中为复现「感知受限环境建模」而新增的感知误差提取与距离相关高斯分布拟合功能，对应论文 Section II-A（Perception Error Modeling）。

## 1. 目标

在 nuScenes 数据集上评估 BEVFusion 模型，将预测的 3D 边界框与真值（ground truth）逐对象比对，提取以下感知误差：

- 位置误差：纵向 `Δx`、横向 `Δy`
- 速度误差：纵向 `Δvx`、横向 `Δvy`

并将每个误差建模为**目标距离相关的高斯分布**：

```
Δx  ~ N(μx(d),  σx²(d))
Δy  ~ N(μy(d),  σy²(d))
Δvx ~ N(μvx(d), σvx²(d))
Δvy ~ N(μvy(d), σvy²(d))
```

其中 `d` 为自车到目标对象的欧氏距离，均值 `μ(d)` 与标准差 `σ(d)` 均用关于 `d` 的二次多项式拟合：

```
μ(d) = a0 + a1·d + a2·d²
σ(d) = b0 + b1·d + b2·d²
```

拟合得到的系数即论文 Table III 的感知噪声模型参数，供后续仿真阶段采样并注入背景车辆状态。

## 2. 修改的文件

共修改 3 个已有文件，并新增 1 个独立脚本：

### 2.1 `tools/eval_utils/eval_utils.py`

新增 3 个函数，并给 `eval_one_epoch` 增加 `extract_error` 参数：

| 函数 | 作用 |
|---|---|
| `_greedy_match_centers()` | 按 BEV 中心距离 + 类别一致做贪心匹配（默认阈值 2.0 m，对齐 nuScenes 匹配惯例） |
| `extract_perception_errors()` | 对每个匹配对计算 `d`、`dx`、`dy`、`dvx`、`dvy` |
| `fit_distance_dependent_gaussian()` | 按距离分 bin（默认 5.0 m），逐 bin 计算 μ 与 σ，用加权最小二乘拟合二次多项式 |
| `_save_perception_error()` | 输出系数表到日志，保存 `perception_error_fit.json` 与 `perception_error_records.npz` |

`eval_one_epoch` 的改动：

- 签名新增 `extract_error=False`。
- 循环内：`if extract_error:` 时累积 `extract_perception_errors(batch_dict, pred_dicts)`。
- 循环后（`cfg.LOCAL_RANK == 0`）：调用 `fit_distance_dependent_gaussian` + `_save_perception_error`。
- 分布式评测（`dist_test`）下使用 `torch.distributed.all_gather_object` 汇总各 rank 的误差记录。

### 2.2 `tools/test.py`

- 新增命令行开关 `--extract_perception_error`（`store_true`，默认 `False`）。
- 将 `extract_error=args.extract_perception_error` 透传给两处 `eval_utils.eval_one_epoch(...)` 调用。

### 2.3 `tools/extract_perception_error.py`（新增）

独立的轻量脚本，只做「推理 → 匹配 → 误差提取 → 拟合 → 输出」，不运行 nuScenes 官方评测。核心逻辑全部复用 `eval_utils` 中的 `extract_perception_errors` / `fit_distance_dependent_gaussian` / `_save_perception_error`，因此与集成方式产生**完全一致**的误差样本与拟合系数。

主要参数：

| 参数 | 作用 |
|---|---|
| `--ckpt` | BEVFusion 权重（`.pth`） |
| `--match_dist_thresh` | 匹配中心距离阈值（m，默认 2.0） |
| `--dist_bin_width` | 拟合分 bin 宽度（m，默认 5.0） |
| `--min_samples_per_bin` | 每 bin 最少样本数（默认 5） |
| `--records_npz` | 提供时跳过推理，直接从已有 npz 重新拟合 |
| `--plot` | 输出 `perception_error_fit.png` 拟合曲线图 |
| `--save_dir` | 输出目录（默认 `output/<exp>/<tag>/perception_error/`） |

### 2.4 `pcdet/datasets/__init__.py`（数据加载修复）

该文件原先在模块顶部无条件 `import` 了全部 8 个数据集。其中 `argo2`（依赖 `av2` + `pandas`）、`waymo`（依赖 `SharedArray`）所需的 SDK 不在 `requirements.txt` 中，任何一个 import 失败都会导致整个 `pcdet.datasets` 报错，连 nuScenes 也无法使用。

修复：将非 nuScenes 的数据集 import 与 `__all__` 条目全部注释掉，仅保留：

```python
from .dataset import DatasetTemplate
from .nuscenes.nuscenes_dataset import NuScenesDataset

__all__ = {
    'DatasetTemplate': DatasetTemplate,
    'NuScenesDataset': NuScenesDataset,
}
```

- `DatasetTemplate` 为基类（`tools/demo.py` 仍引用），必须保留。
- `build_dataloader` 通过 `__all__[dataset_cfg.DATASET]` 查找 `NuScenesDataset`，保留即可正常加载。

## 3. 数学定义与代码映射

### 3.1 坐标系与数据格式

nuScenes 在 OpenPCDet 中使用 LiDAR 坐标系，其中 x 轴为纵向（前向）、y 轴为横向（左侧），ego 位于原点。测试模式下无数据增强，`gt_boxes` 与 `pred_boxes` 均在 LiDAR 坐标系。

| 数据 | 列布局 |
|---|---|
| `gt_boxes`（10 列） | `[x, y, z, dx, dy, dz, heading, vx, vy, class]` |
| `pred_boxes`（9 列） | `[x, y, z, dx, dy, dz, heading, vx, vy]` |

（`pred_boxes` 的 9 列由 `transfusion_head.py` 中 `torch.cat([center, height, dim, rot, vel])` 得到，`vel` 位于第 7、8 列。）

### 3.2 误差计算

对每个匹配对 `(gt_i, pred_i)`：

```python
d   = hypot(gt_x, gt_y)          # 自车到目标的 2D 欧氏距离（取真值位置）
dx  = pred_x  - gt_x             # 纵向位置误差
dy  = pred_y  - gt_y             # 横向位置误差
dvx = pred_vx - gt_vx            # 纵向速度误差
dvy = pred_vy - gt_vy            # 横向速度误差
```

### 3.3 拟合

1. 以 `bin_width = 5.0 m` 对 `d` 分 bin。
2. 每个 bin 内计算样本均值 `μ` 与样本标准差 `σ`（`ddof=1`）。
3. 用加权最小二乘 `np.polyfit(d_centers, means, 2, w=sqrt(counts))` 拟合 `μ(d)`，得到 `[a2, a1, a0]`（`np.polyfit` 返回高次在前）。
4. 同理拟合 `σ(d)`，得到 `[b2, b1, b0]`。

## 4. 运行方式

### 4.1 独立脚本（推荐，跳过官方评测）

```bash
python tools/extract_perception_error.py \
  --cfg_file tools/cfgs/nuscenes_models/bevfusion.yaml \
  --ckpt /path/to/bevfusion_checkpoint.pth \
  --plot
```

- 该脚本只做「推理 → 匹配 → 误差提取 → 拟合 → 输出」，不运行 nuScenes 官方评测，因此**不依赖 `nuscenes-devkit`**、也不需要 GT 评测集。
- `--plot` 可选，输出 `perception_error_fit.png`（μ/σ 随距离的分 bin 点 + 拟合曲线）。
- 若已生成 `perception_error_records.npz`，可跳过推理，只重新拟合（便于调 `--dist_bin_width`、`--min_samples_per_bin` 等参数，速度快）：

```bash
python tools/extract_perception_error.py \
  --cfg_file tools/cfgs/nuscenes_models/bevfusion.yaml \
  --records_npz output/.../perception_error/perception_error_records.npz \
  --save_dir output/.../perception_error
```

### 4.2 集成到标准评测

```bash
python tools/test.py \
  --cfg_file tools/cfgs/nuscenes_models/bevfusion.yaml \
  --ckpt /path/to/bevfusion_checkpoint.pth \
  --extract_perception_error
```

该方式在标准评测基础上额外提取误差，但会同时运行 nuScenes 官方评测（需 `nuscenes-devkit` 与 GT）。

> 两种方式（4.1 与 4.2）生成的误差样本与拟合系数**完全一致**——它们复用同一套 `extract_perception_errors` / `fit_distance_dependent_gaussian` 函数、同一模型、同一份数据；官方评测步骤只影响 mAP/NDS 指标，不改变误差提取结果。仅需生成分布时推荐用 4.1。

## 5. 输出

结果目录（两种方式，内容一致）：

- 独立脚本：`output/<exp_group_path>/<tag>/perception_error/`
- 集成方式：`output/<exp_group_path>/<tag>/<extra_tag>/eval/epoch_<N>/<split>/perception_error/`

| 文件 | 内容 |
|---|---|
| `perception_error_fit.json` | 每个变量（`dx`/`dy`/`dvx`/`dvy`）的 `mu{a0,a1,a2}`、`sigma{b0,b1,b2}` 及分 bin 统计 |
| `perception_error_records.npz` | 全部匹配对的原始误差样本：`d`、`dx`、`dy`、`dvx`、`dvy` |
| `perception_error_fit.png` | （仅 `--plot` 时）μ/σ 随距离的分 bin 点 + 拟合曲线 |

日志中会打印形如下的系数表（即论文 Table III）：

```
    var           a0           a1           a2 |           b0           b1           b2
--------------------------------------------------------------------------------
     dx      0.000000    0.000000    0.000000 |    0.000000    0.000000    0.000000
     dy      0.000000    0.000000    0.000000 |    0.000000    0.000000    0.000000
    dvx      0.000000    0.000000    0.000000 |    0.000000    0.000000    0.000000
    dvy      0.000000    0.000000    0.000000 |    0.000000    0.000000    0.000000
```

## 6. 实现说明与假设

- **匹配策略**：贪心匹配，要求类别一致且 BEV 中心距离 ≤ 2.0 m。这是对论文「predicted 3D bounding boxes 与 ground-truth 逐一比对」的一种标准实现（对齐 nuScenes 中心距离匹配惯例）。
- **距离 d**：取真值位置的 2D 欧氏距离 `√(x²+y²)`，与仿真阶段「自车与背景车的欧氏距离」一致；忽略高度 z。
- **聚合范围**：当前对所有类别（含行人、自行车等）一起提取与拟合，符合论文「statistically aggregate diverse challenging scenes」的描述；若只需车辆类，可在 `extract_perception_errors` 中按类别过滤。
- **拟合方法**：分 bin + 加权最小二乘二次拟合，是论文「data fitting procedures confirmed quadratic polynomials」的直接实现；`bin_width`、`min_samples_per_bin` 为可调参数（函数默认值）。

## 7. 已知依赖注意

- `SharedArray` 在 `pcdet/utils/common_utils.py`（第 7 行）与 `pcdet/datasets/augmentor/database_sampler.py`（第 8 行）中被顶层无条件 `import`，是**全局硬依赖**（在 `requirements.txt` 中）。它是易装不上的 C 扩展；若安装失败会报 `ModuleNotFoundError: No module named 'SharedArray'`，此时需先 `pip install SharedArray`。本仓库当前**未**将其改为可选/懒加载。

---

# 第二部分：感知受限场景生成（sce_gen）

> 本部分记录 `sce_gen/` 目录下 RL 场景生成环境的全部改动。它消费第一部分拟合出的感知噪声系数（Table III），
> 在自车对周围车辆的观测上注入距离相关的高斯噪声，用 RL 训练对抗背景车制造「感知受限下的危险场景」。

## 8. 文件清单

| 文件 | 说明 |
|---|---|
| `sce_gen_ori.py` | 原始版本（保留作备份，未修改） |
| `sce_gen.py` | 当前 CPU 版本（本文档多数改动在此落地） |
| `sce_gen_gpu.py` | GPU 训练版本（`sce_gen.py` 的 argparse/device 包装） |
| `sce_gen_parallel.py` | 真并行版本（SubprocVecEnv 多进程并行采样） |
| `sce_gen_en.py` | `sce_gen.py` 的英文注释版（新文件） |
| `sce_gen_gpu_en.py` | `sce_gen_gpu.py` 的英文注释版（新文件） |
| `submit_sce_gen.py` | 提交 DLP 集群的脚本 |
| `analyze_results.ipynb` | 多 seed 结果合并分析 notebook（标题/标签已英文） |
| `analyze_results_en.ipynb` | 上述 notebook 的全英文版（新文件） |

## 9. 车辆角色定义（关键）

| id | 角色 | 控制方式 | 观测 |
|---|---|---|---|
| 0 | 自车（被测试系统） | IDM + MOBIL（可换道） | **带感知噪声**的周围车辆 |
| 1 | 对抗背景车 | RL（纵向加速度 + 横向位移） | **真实状态**（无感知受限） |
| 2,3,4 | 普通背景车 | IDM + MOBIL | 真实状态 |

- 自车不再由 RL 控制，改为正常驾驶（IDM + MOBIL），是被测系统。
- RL 智能体是**对抗背景车**（`adversary_id = 1`），负责制造危险场景。
- 感知噪声只影响**自车看周围其他车**：自车对自身、以及背景车之间均为真实感知。
- 对应代码：`Car.__init__` 新增 `is_adversary` 字段；`reset()` 里 `adversary_id`；`step()` 分支改为
  `if car.is_adversary: ... elif car.is_ego: ... else: ...`。

## 10. 感知噪声模型（距离依赖二次高斯）

`perception_noise_coeffs`（第一部分拟合的 Table III 系数，硬编码在代码里）：

```
mu(d)    = a0 + a1*d + a2*d^2
sigma(d) = b0 + b1*d + b2*d^2
```

四维误差 `(dx, dy, dvx, dvy)` 各有一组 `(mu, sigma)` 系数，与第一部分的 `perception_error_fit.json` 一致。

- `_sample_perception_noise(d)`：按距离 `d` 采样噪声（`sigma` 裁剪为非负）。
- `_update_perception()`：每步采样一次自车的带噪感知并缓存到 `last_perceived_states`，供决策与日志共用同一次噪声。
- `_ego_perceived_cars()`：返回自车眼中的车辆列表（自车用真实自身，其余用带噪值），供自车 IDM/MOBL 决策使用（浅拷贝，不改真实状态）。
- `enable_perception_noise = True` 开关，消融实验可设为 `False`。

## 11. TTC 计算修复

修复了 `step()` 中前向追尾 TTC 被 `pass` 跳过、只算后车追尾的 bug：

```python
if abs(agent.lane_pos - npc.lane_pos) < 2.2:
    if dist_x > 0 and rel_v > 0:
        # 自车追上前方慢车（前向追尾风险）
        ttc = max((dist_x - agent.length), 0.0) / rel_v
        min_ttc = min(min_ttc, ttc)
    elif dist_x < 0 and rel_v < 0:
        # 后车追上自车
        ttc = max((abs(dist_x) - agent.length), 0.0) / abs(rel_v)
        min_ttc = min(min_ttc, ttc)
```

与 `_simulate_future_ttc` 里的代理模型计算保持一致。

## 12. 奖励函数（对齐论文公式）

奖励 = 安全临界奖励 + 感知风险奖励：

- **`r_safety`（论文公式 16）**：`TTC_LOW=1.0`、`TTC_HIGH=4.0`、`alpha_safety=5.0`、`beta_safety=1.0`
  - `min_ttc < TTC_LOW` 或碰撞 → `beta_safety`
  - `TTC_LOW <= min_ttc <= TTC_HIGH` → `alpha_safety * (TTC_HIGH - min_ttc) / (TTC_HIGH - TTC_LOW)`
  - 其余 → 0（到最大步数则 truncated）
- **`r_perception`（论文公式 17 / ERD）**：`_compute_perception_risk_reward`
  - 用真实观测 + 带噪观测分别前向仿真 `n_steps` 步、Monte Carlo `n_samples` 次
  - 风险度量 `1/TTC`，`Delta = E[R_obs] - E[R_true]`
  - 奖励 = `alpha_perc * |Delta| + beta_perc * max(0, -Delta)`
- 综合奖励 `reward = r_safety + r_perception`，裁剪到 `[-10, 100]`。
- `info` 里新增 `r_safety`、`r_perception`、`perception_delta` 字段。

## 13. HDF5 日志扩展

每个 `episode_*` 组新增 `perception_data` 数据集，schema：

- `trajectories` `(T, 5, 4)`：`[纵向位置 pos, 横向位置 lane_pos, 纵向速度 speed, 0]`，车顺序 `[ego, adversary, bg2, bg3, bg4]`
- `perception_data` `(T, 5, 13)`：`[感知值(4) | 真实值(4) | 误差 delta(4) | 距离(1)]`
  - 感知值：`[perceived_pos, perceived_lane_pos, perceived_speed, perceived_vy]`
  - 真实值：`[true_pos, true_lane_pos, true_speed, true_vy]`
  - 误差：`[dx, dy, dvx, dvy]`
  - 距离：`dist_to_ego`
- attrs：`collision`、`episode_length`、`episode_reward`。

## 14. GPU 版本 `sce_gen_gpu.py`

基于 `sce_gen.py` 新增：

- argparse：`--gpu`（默认 0）、`--timesteps`（默认 200000）、`--seed`（默认 0）
- `device = f'cuda:{args.gpu}'`
- `record_path = f"./ppo_logs_gpu{args.gpu}_{timestamp}/"`
- PPO 传入 `device=device, seed=args.seed, tensorboard_log=record_path + "tensorboard"`
- `model.learn(total_timesteps=args.timesteps)`；`model.save(record_path + f"ppo_{args.timesteps // 1000}k")`

> 注：`sce_gen.py`（CPU 版）原先硬编码 tensorboard 路径 `"/root/autodl-tmp/OpenPCDet/ppo_logs/"`（autodl 专用）。
> 现已修复为 `tensorboard_log=record_path + "tensorboard"`，与本地 `./ppo_logs_{timestamp}/` 目录一致，不再依赖 autodl 路径。

## 15. 提交脚本 `submit_sce_gen.py`

参考 GoalFlow 的 DLP 提交方式（`from dlp_common import setup, submit, make_task_name, WORKSPACE_ROOT`）。

- 多卡语义：**独立并行**（每张卡一个独立进程 + 独立 seed），非数据并行 DDP。SB3 PPO 无原生 DDP；瓶颈在 CPU 环境采样。
- `PIP_PACKAGES = ["stable_baselines3", "gymnasium", "h5py"]`
- `build_launcher` 生成 bash 脚本，`for i in seq 0 (N-1)` 启动
  `python3 sce_gen_gpu.py --gpu $i --timesteps T --seed $((base+i))`，日志重定向到 `gpu_$i.log`。
- 用法示例：

  ```bash
  python3 submit_sce_gen.py --resource-num 4 --timesteps 200000 --seed 42
  ```

## 16. 分析 notebook `analyze_results.ipynb`

合并多个 seed 的危险场景库，输出统计、分布与可视化。头部 `DATA_DIR` / `H5_GLOB` 改路径即可复用。

- 总体统计 + 分 run 对比（碰撞率 / 危险率 / 长度 / 奖励）
- 分布：长度、TTC(log)、奖励直方图；分 run 柱状图 + boxplot
- 感知误差分析：误差分布 + 误差随距离变化，叠加拟合 `mu(d)` / `mu±sigma(d)` 曲线
- ERD-lite：感知 TTC vs 真实 TTC（对角线上方 = 低估危险）+ 风险低估直方图
- 样例可视化：碰撞场景 / 最小 TTC 场景的纵向、横向位置时间曲线

## 17. 训练结果（任务 `n260827-174817-scegen-ppo`）

- 4 × H20 GPU，每卡 200k 步（合计 800k 步，独立 seed）
- 结果目录：`/workspace/asw-shared/dlp/training_tasks/aoh6szh/n260827-174817-scegen-ppo/`
- 每卡一个 `ppo_logs_gpu{0..3}_*/vae-ppo_vehicle_trajectories_*.h5`，每个约 2500 episode
- 合并统计（4 seed 共 9988 场景）：碰撞率 3.06%、危险（TTC<4s）11.90%、严重（TTC<1s）4.55%
- 感知风险低估：真实危险时刻中 54.17% 自车低估了危险（感知 TTC > 真实 TTC）
- 已知警告：CUDA 驱动偏旧（`found version 12020`），训练可正常完成；若报错可改用
  `DLP_GPU_TYPE=H20-CUDA13.0-R4T2`。

## 18. 英文注释版（`sce_gen_en.py` / `sce_gen_gpu_en.py`）

为英文展示/交流用途，将 `sce_gen.py` 与 `sce_gen_gpu.py` 全文翻译为英文注释版：

- `sce_gen_en.py`：`sce_gen.py` 的英文版（类名、函数名、逻辑不变，仅注释/docstring/print 英文）。
- `sce_gen_gpu_en.py`：`sce_gen_gpu.py` 的英文版（由 `sce_gen_en.py` 经同样的 argparse/device 变换生成）。

两版功能与中文版**完全等价**，仅语言不同；`sce_gen_en.py` 同样已修复 autodl 路径问题（`tensorboard_log=record_path + "tensorboard"`）。

## 19. 真并行版本 `sce_gen_parallel.py`

基于 `sce_gen_gpu.py` 改造，用 `SubprocVecEnv` 实现**多进程并行采样**（区别于 15 节的「多卡独立并行」）：

- `DummyVecEnv` → `SubprocVecEnv`（多进程，每个 worker 独立 GIL，真正并行 CPU 采样）。
- 新增 `--num-envs` 参数，默认 `min(os.cpu_count() or 1, 16)`。
- `record_path = f"./ppo_logs_parallel_{timestamp}/"`。
- 每个 worker 独立 HDF5 文件 + 独立随机种子：

```python
def make_env(rank, base_seed, record_path, timestamp):
    def _init():
        np.random.seed(base_seed + rank)   # 每个 worker 显式设置独立种子
        filename = record_path + f"vae-ppo_vehicle_trajectories_worker{rank}_{timestamp}.h5"
        return BasePhysicsEnv(output_file=filename)
    return _init

vec_env = SubprocVecEnv(
    [make_env(i, args.seed, record_path, timestamp) for i in range(args.num_envs)]
)
```

**与之前版本的一致性**：

- 环境类 `BasePhysicsEnv`、奖励函数、感知噪声模型、HDF5 schema **完全不变** → 产出的危险场景分布与之前**统计一致**。
- 但**非逐位一致**：
  1. 每个 worker 用 `np.random.seed(base_seed + rank)`，采样出的具体轨迹与单环境版不同（同分布、不同样本）。
  2. `total_timesteps=200000` 是**所有环境加总**，每个 worker 只跑 `200000 / N` 步，单个 worker 文件 episode 数约为之前的 `1/N`（全部 worker 合计大致持平）。

**注意事项**：

- 每个 worker 进程约 300–500 MB 内存；worker 数不要超过物理 CPU 核数。
- DLP 共享网络盘上多进程同时写 HDF5 可能有磁盘 IO 瓶颈，需实测 FPS 调 `--num-envs`。
- 若需严格复现某次结果（固定 seed 出同一批数据），把 `--num-envs` 设为 1 即可。

## 20. 分析 notebook 英文版

- `analyze_results.ipynb`：中文本体、**英文图标题/标签**（供英文展示）。
- `analyze_results_en.ipynb`：全英文版（markdown/code/print/图标题/标签全部英文，无 CJK）。

并行版产物目录为 `ppo_logs_parallel*`，复用时将 notebook 头部 `H5_GLOB` 改为对应目录即可。


# sce_gen_gpu_en.py 改动说明

## 概述

本文档记录了对 `sce_gen_gpu_en.py` 的改动，使其成为一个**考虑感知受限的自动驾驶测试场景生成系统**。核心目标是：用 PPO 控制对抗背景车，对自车产生对抗行为，生成**真实且多样、具有实际价值的 near-miss 场景**，而非碰撞场景。

---

## 核心架构设计

### 场景生成逻辑

1. **对抗背景车（RL 控制）**：由 PPO 策略直接控制纵向加速度和横向偏移，目的是制造危险但可避免的场景
2. **自车（Ego）**：使用 IDM + MOBIL 控制，**感知受限**——感知到的周围车辆位置/速度含有距离相关的高斯噪声
3. **其他背景车**：使用 IDM + MOBIL 控制，**感知完全**（看到真实状态）
4. **碰撞判据**：使用真实物理状态（无噪声）

```
真值物理引擎 → 自车感知加噪 → 自车基于噪声决策 → 真值判碰撞/奖励
```

---

## 奖励函数设计（关键改动）

### 问题：原始设计的缺陷

**原有问题**：r_safety 随 TTC 减小而单调递增 → 策略最优解是"悬停在 TTC≈1 但绝不撞"，或直接碰撞，**不会生成可持续的 near-miss 场景**。

### 新设计：平台型 near-miss 奖励

```python
# Near-miss target band
TTC_NEAR_LOW  = 0.8   # 低于此值太接近碰撞 → 惩罚
TTC_NEAR_HIGH = 2.5   # 高于此值不足够危险 → 无奖励
TTC_SAFE      = 5.0   # 超出此值完全安全 → 无奖励
R_NEAR_MISS   = 5.0   # 目标带内的奖励
```

奖励曲线：

```
TTC:   0.0   0.4   0.8   1.5   2.5   3.0   4.0   5.0   100
r:    -5.0   0.0  +5.0  +5.0  +5.0  +4.0  +2.0   0.0   0.0
                  └──── 目标带 ────┘
```

### 碰撞惩罚（不可薅羊毛）

- `COLLISION_PENALTY = -50.0`
- 在 clip 之外施加 → 即使提前积累了高额奖励，碰撞仍会导致总收益为负
- 与 `R_NEAR_MISS = 5.0` 对比：需要 10 步持续 near-miss 才能抵消一次碰撞

### 生存奖励（鼓励持续施压）

- `SURVIVAL_BONUS = 25.0`
- 只有跑完整个 episode 且不碰撞才发放
- 按 `带内时间占比` 加权 → "持续施压但始终不接触"成为最优策略

### 自然性惩罚（保证真实性）

防止对抗车产生 bang-bang 急刹或疯狂画龙等物理合法但不现实的行为：

```python
r_natural = -[w_jerk * (jerk/10)^2 + w_lat * (lat_change/2.5)^2 + w_harsh * max(0, -acc-3)^2]

w_jerk = 0.30   # 纵向加加速度
w_lat  = 0.50   # 横向摆动
w_harsh = 0.20  # 持续急刹（<-3 m/s²）
```

标定示例：

| 行为 | 惩罚/步 |
|------|---------|
| 平稳巡航 | -0.02 |
| 激进但真实的切入 | -0.17 |
| 随机 bang-bang + 画龙 | -4.9 |

---

## 感知噪声模型改进

### 问题：原始二次拟合在远距离失效

原始 sigma(d) = a0 + a1*d + a2*d²（a2<0）在 d≈40-50m 达峰后回落，在 d≈90-105m 过零 clamp 到 1e-6 → **远车反而观测得完美**，物理上反了。

### 解决：sigma 饱和在峰值处

```python
self.sigma_saturation_dist[var] = -b1 / (2*b2)  # 二次函数驻点
d_sigma = min(d, self.sigma_saturation_dist[var])
```

验证输出（以 x 轴为例）：
```
sigma_x @ d=20/45/90/200 : [0.1166, 0.1519, 0.1519, 0.1519]
```

---

## 观测空间改进

### 问题：横向速度 vy 恒为 0

`get_true_state()` 第 4 维硬编码为 0.0，即使对抗车正在横移。

### 解决：新增 `Car.lat_speed`

```python
# Car.__init__ 新增
self.lat_speed = 0.0

# step() 中每步更新
lane_pos_before = car.lane_pos
# ... physics update ...
car.lat_speed = (car.lane_pos - lane_pos_before) / time_step

# get_true_state() 使用
state.append([car.pos - base_x, car.lane_pos, car.speed, car.lat_speed])
```

---

## 对抗车生成位置优化

### 问题：大量 episode 中对抗车生成在 150m 外，根本够不着自车

### 解决：对抗车初始位置约束

```python
if i == self.adversary_id:
    offset = np.random.uniform(10.0, 50.0) * np.random.choice([-1.0, 1.0])
    pos = ego.pos + offset
else:
    pos = np.random.uniform(0, 200)
```

---

## 稳定性与调试改进

### HDF5 文件处理

```python
# 每 100 步 flush
if self.current_step % 100 == 0:
    self.h5_file.flush()

# 新增 close() 方法
def close(self):
    if getattr(self, 'h5_file', None) is not None:
        try:
            self.h5_file.flush()
            self.h5_file.close()
        finally:
            self.h5_file = None
```

### 代码清理

- 删除无用的 `decide_lane_change_ego()` 方法（含拼写错误 `self.lane_change_progress`）

---

## 场景可筛选性

每个 episode 的 HDF5 attrs 现在包含：

```python
{
    'collision': bool,
    'episode_length': int,
    'episode_reward': float,
    'min_ttc': float,
    'near_miss_steps': int,      # 新增
    'is_near_miss': bool         # 新增
}
```

`info` 字典新增字段：

- `r_natural`：自然性惩罚
- `near_miss`：本步是否在目标带内

---

## 参数配置表

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `TTC_NEAR_LOW` | 0.8 | near-miss 下限 |
| `TTC_NEAR_HIGH` | 2.5 | near-miss 上限 |
| `TTC_SAFE` | 5.0 | 完全安全阈值 |
| `R_NEAR_MISS` | 5.0 | 目标带内奖励 |
| `COLLISION_PENALTY` | -50.0 | 碰撞惩罚 |
| `SURVIVAL_BONUS` | 25.0 | 生存奖励 |
| `w_jerk` | 0.30 | jerk 惩罚系数 |
| `w_lat` | 0.50 | 横向摆动惩罚系数 |
| `w_harsh` | 0.20 | 急刹惩罚系数 |
| `alpha_safety` | 5.0 | 安全奖励系数（保留备用） |
| `beta_safety` | 1.0 | 安全奖励系数（保留备用） |
| `alpha_perc` | 2.0 | 感知风险奖励系数 |
| `beta_perc` | 4.0 | 感知风险奖励系数 |

---

## 文件修改概览

| 位置 | 修改内容 |
|------|----------|
| `__init__` | 新增 12 行参数；新增 `sigma_saturation_dist` 初始化 |
| `_poly2`, `_sample_perception_noise` | 修改 noise sampling 使用饱和距离 |
| `_near_miss_reward` | 新增方法（35 行） |
| `_naturalness_penalty` | 新增方法（20 行） |
| `reset` | 新增 episode 初始状态初始化 |
| `step` | 修改 reward 计算逻辑；新增 per-step near-miss tracking；新增碰撞后惩罚；新增生存奖励 |
| `close` | 新增方法 |
| Car class | 新增 `lat_speed` 属性 |
| 训练脚本末尾 | 新增 `raw_env.close()` |

---

## 已知保留项（按需求）

- **感知噪声时间相关性（AR 模型）**：未实现，按需求保留白噪声

---

## 参考资料

- IDM 模型：`Car.calculate_acceleration()`
- MOBIL 换道：`Car.evaluate_lane_change()`, `Car.decide_lane_change()`
- 感知噪声模型：`_sample_perception_noise()`, `_ego_perceived_cars()`
- 近期碰撞风险估计：`_compute_perception_risk_reward()`

● The visualization script now works end-to-end. Summary of what I fixed and produced:

  Fixed (visualize_scenario.py):
  1. inf-TTC crash — capped inf TTC values with TTC_CAP=100.0 and capped the underestimation score in select_episode
  (this was the earlier Axis limits cannot be NaN or Inf crash).
  2. Portrait blowup — the bird's-eye view originally drew the road vertically (longitudinal = y-axis), which combined
  with aspect='equal' forced a 1783×2912 portrait. I rotated it to the conventional AV layout: ego drives right, x = 
  forward distance, y = lateral position, so the road runs horizontally and the figure stays landscape.
  3. TTC panel legibility — the y-axis was auto-scaling to 100+ (dominated by "no conflict" frames), hiding the 0–4s
  danger window. Added TTC_YLIM=15.0 with clipped display so the danger/risk bands are visible.
  4. Added a shaded road surface, risk-band shading, and kept the dashed "ghost" perception boxes + true↔perceived
  offset lines.

  Selected scenario (textbook perception-limited case):
  episode_80  collision=True  min_true_TTC=1.37s  underestimation_at_danger=98.63s
  At the danger frame the ego's perceived TTC is 100s ("no conflict") while the true TTC is 1.37s — the ego completely
  missed the imminent collision.

  Outputs (in /workspace/aoh6szh/ai_planning_research/sce_gen/):
  - typical_scenario.png — static image at the most dangerous moment (2100×1200)
  - typical_scenario.mp4 — full-episode animation (10 fps, ~1 MB)

  I confirmed via pixel analysis that the PNG contains the road/lanes, ego (blue), adversary (red), and the yellow
  collision marker — but note I can't render images in this environment, so I couldn't visually eyeball the aesthetics
  myself.

  To re-run or tweak: edit DATA_DIR / H5_GLOB at the top, adjust VIEW_AHEAD/VIEW_BEHIND/TTC_YLIM, then python3 
  visualize_scenario.py.

