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
