import torch
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from torch.distributions import Normal
import gymnasium as gym
import time
import matplotlib.pyplot as plt
import math
import h5py
from gymnasium import spaces

class DummySpec:
    id = "MOTrafficEnv-v0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置仿真参数
lane_width = 4.0  # 每车道宽度（米）
time_step = 0.2  # 每步仿真时间（秒）
lanes = 3  # 三车道
total_time = 30  # 仿真总时长（秒）
collision_threshold = 2.0  # 碰撞阈值（米）
num_cars = 5  # 车流密度（辆）
# NDD Vehicle IDM parameters
COMFORT_ACC_MAX = 2 # [m/s2]
COMFORT_ACC_MIN = -4.0  # [m/s2]
DISTANCE_WANTED = 5.0  # [m]
TIME_WANTED = 1.5  # [s]
DESIRED_VELOCITY = 35 # [m/s]
DELTA = 4.0  # []
# CAV surrogate model IDM parameter
SM_IDM_COMFORT_ACC_MAX = 2.0  # [m/s2]  2
SM_IDM_COMFORT_ACC_MIN = -4.0  # [m/s2]  -4
SM_IDM_DISTANCE_WANTED = 5.0  # [m]  5
SM_IDM_TIME_WANTED = 1.5  # [s]  1.5
SM_IDM_DESIRED_VELOCITY = 35 # [m/s]
SM_IDM_DELTA = 4.0  # []
lanechange={'stay':0,'left':1,'right':-1}

class Car:
    def __init__(self, id, lane, pos, lane_pos, speed, length=5, width=2, lanes=3, is_ego=0):
        self.id = id
        self.lane = lane
        self.pos = pos
        self.speed = speed
        self.lane_offset = 0  # 横向偏移量（车道中心偏移）
        self.length = length #车辆长度
        self.width = width  # 车辆宽度
        self.change_lane_progress = 0  # 换道进度
        self.lane_change_direction = ''
        self.lanes = lanes
        
        # 修复 2: 直接使用传入的 lane_pos，而不是强行计算
        self.lane_pos = lane_pos 
        
        self.is_ego = is_ego  # 是否为强化学习控制的智能体车辆
        self.acceleration = 0  # 默认加速度为0
        self.leading_distance = -1  # 默认前车距离为-1
    def calculate_acceleration(self, leading_car):
        a0 = COMFORT_ACC_MAX
        v0 = SM_IDM_DESIRED_VELOCITY
        delt = SM_IDM_DELTA
        acceleration = a0 * (1 - np.power(self.speed/v0, delt))
        if leading_car is not None:
            r = leading_car.pos-self.pos
            d = max(1e-5, r - self.length)
            d0 = DISTANCE_WANTED
            tau = TIME_WANTED
            ab = -COMFORT_ACC_MAX * COMFORT_ACC_MIN
            dv = self.speed - leading_car.speed
            d_star = d0 + max(0, self.speed * tau +self.speed * dv / (2 * np.sqrt(ab)))
            acceleration -= a0 * np.power(d_star/ d, 2)
        if acceleration<-8:
            acceleration=-8
        return acceleration
    
    def update_position(self, acceleration):
        """
        更新车辆的纵向位置，使用IDM模型控制加速度。
        :param acceleration: 车辆的纵向加速度
        """
        self.pos += self.speed*time_step+0.5*acceleration*time_step**2
        self.speed += acceleration*time_step
        if self.speed<0:
            self.speed=0    
        
    
    def update_lane_position(self):
        """
        换道时平滑处理横向位移
        """
        if self.change_lane_progress > 0 and self.change_lane_progress <= 5:
            # 换道的横向位移量，左换道为正值，右换道为负值
            delta_offset_list =[i*lane_width for i in [0.1, 0.2, 0.4, 0.2, 0.1]] # 换道的横向位移量
            delta_offset=delta_offset_list[self.change_lane_progress - 1]
            #print('0')
            if self.lane_change_direction == "right":
                self.lane_offset -= delta_offset
                self.lane_pos-=delta_offset
                self.lane=self.get_lane_from_position()
                #print('1')
            else:  # 如果是左换道
                self.lane_offset += delta_offset
                self.lane_pos+=delta_offset
                self.lane=self.get_lane_from_position()
                #print('2')
            self.change_lane_progress += 1  # 换道进度增加
        # 完成换道后，重置进度
        if self.change_lane_progress > 5:
            self.change_lane_progress = 0  # 完成换道后进度归零
        
        # 当换道进度为负时，尽快回到最近车道中间
        if self.change_lane_progress < 0:
            target_lane_center = (self.lane - 0.5) * lane_width  # 当前车道的中心位置
            lateral_diff = target_lane_center - self.lane_pos  # 横向偏移量
            max_lateral_move = lane_width / 4  # 最大横向位移量为四分之一车道宽

            # 根据最大位移量调整横向位置
            if abs(lateral_diff) <= max_lateral_move:
                self.lane_pos = target_lane_center  # 直接移动到车道中心
                self.change_lane_progress = 0  # 重置换道进度
            else:
                self.lane_pos += max_lateral_move if lateral_diff > 0 else -max_lateral_move
                self.lane = self.get_lane_from_position()  # 更新车道编号
    
    def update_lane_offset(self):
        """
        更新车辆相对于最近车道中心线的偏移量
        """
        # 计算最近车道中心线的位置
        target_lane_center = (self.lane - 0.5) * lane_width
        # 计算车辆相对于车道中心线的偏移量
        self.lane_offset = self.lane_pos - target_lane_center

    def get_lane_from_position(self):
        lane_id = int(self.lane_pos // lane_width)+1
        return lane_id 
    
    def get_lateral_range(self, lane_width=4):
        """计算车辆当前横向覆盖范围 [start, end]"""
        actual_lateral = self.lane_pos
        return (actual_lateral - self.width/2, actual_lateral + self.width/2)
    
    def is_overlapping(self, other_car, lane_width=4):
        """检测与另一车辆的横向范围是否重叠"""
        s_start, s_end = self.get_lateral_range(lane_width)
        o_start, o_end = other_car.get_lateral_range(lane_width)
        return (s_start < o_end) and (s_end > o_start)
    
    def find_leading_car(self, cars):
        """在当前横向范围内，找到最近的前车"""
        leading = None
        for car in cars:
            if car.id == self.id or not self.is_overlapping(car) or car.pos <= self.pos:
                continue
            if leading is None or car.pos < leading.pos:
                leading = car
        self.leading_distance = leading.pos - self.pos if leading else -1
        return leading
    
    # --- MOBIL 换道评估函数 ---
    def evaluate_lane_change(self, target_lane, cars, lane_width=4, b_safe=4.0, p=0.1):
        """评估换道到目标车道的安全性和激励值
        Returns: (是否安全, 激励值)
        """
        # 1. 计算换道后的横向覆盖范围
        target_center = (target_lane - 0.5) * lane_width
        new_lateral_start = target_center - self.width/2-lane_width/4
        new_lateral_end = target_center + self.width/2+lane_width/4

        # 2. 在目标车道中寻找最近的前车和后车
        leading, following = None, None
        for car in cars:
            if car.id == self.id:
                continue
            # 检测车辆是否在目标车道横向范围内
            c_start, c_end = car.get_lateral_range(lane_width)
            if (new_lateral_start < c_end) and (new_lateral_end > c_start):
                if car.pos > self.pos:  # 候选前车
                    if leading is None or car.pos < leading.pos:
                        leading = car
                else:  # 候选后车
                    if following is None or car.pos > following.pos:
                        following = car

        # 3. 安全性检查：目标车道后车是否能安全制动
        safe = True
        if following:
            # 计算换道后后车的加速度（假设后车的前车变为自己）
            a_back_new = following.calculate_acceleration(leading_car=self)
            safe = (a_back_new >= -b_safe)

        # 4. 计算激励值（MOBIL 核心公式）
        # 当前自身加速度（原车道）
        current_leader = self.find_leading_car(cars)
        a_current = self.calculate_acceleration(current_leader)

        # 换道后的自身加速度（目标车道）
        a_new = self.calculate_acceleration(leading) 
        #if leading else self.get_free_acceleration()

        # 原车道后车的加速度（如果存在）
        current_follower = None
        for car in cars:
            if car.id == self.id or not self.is_overlapping(car) or car.pos >= self.pos:
                continue
            if current_follower is None or car.pos > current_follower.pos:
                current_follower = car
        a_old_back = current_follower.calculate_acceleration(self) if current_follower else 0

        # 目标车道后车的新加速度（如果存在）
        a_new_back = following.calculate_acceleration(self) if following else 0

        # MOBIL 激励公式：a_new - a_current + p*(a_new_back - a_old_back) > threshold
        incentive = (a_new - a_current) + p * (a_new_back - a_old_back)

        return safe, incentive

    # --- 换道决策函数 ---
    def decide_lane_change(self, cars, max_lane=3, lane_width=4, b_safe=4.0, p=0.1, threshold=1/num_cars):
        """MOBIL 换道决策核心逻辑
        Returns: 'stay', 'left' 或 'right'
        """
        if self.change_lane_progress > 0 or self.change_lane_progress < 0:
            return self.lane_change_direction  # 换道过程中不重复决策

        current_lane = self.lane
        best_action = 'stay'

        max_incentive = threshold  # 激励必须超过阈值
    
        
        # 检查右车道
        if current_lane > 1:
            safe_right, incentive_right = self.evaluate_lane_change(
                current_lane - 1, cars, lane_width, b_safe, p
            )
            if safe_right and incentive_right > max_incentive+0.1:
                max_incentive = incentive_right
                best_action = 'right'
                self.change_lane_progress=1
        
        # 检查左车道
        if current_lane < max_lane:
            safe_left, incentive_left = self.evaluate_lane_change(
                current_lane + 1, cars, lane_width, b_safe, p
            )
            if safe_left and incentive_left > max_incentive:
                max_incentive = incentive_left
                best_action = 'left'
                self.change_lane_progress=1

        return best_action
    
    def decide_lane_change_ego(self, cars, max_lane=3, lane_width=4, b_safe=4.0, p=0.1, threshold=0.2):
        """MOBIL 换道决策核心逻辑
        Returns: 'stay', 'left' 或 'right'
        """
        if self.change_lane_progress > 4:
            self.lane_change_progress=0
        
        if self.change_lane_progress > 0:
            return lanechange[self.lane_change_direction] # 换道过程中不重复决策

        current_lane = self.lane
        best_action = 0
        max_incentive = threshold  # 激励必须超过阈值
        
        # 检查右车道
        if current_lane > 1:
            safe_right, incentive_right = self.evaluate_lane_change(
                current_lane - 1, cars, lane_width, b_safe, p
            )
            if safe_right and incentive_right > max_incentive+0.1:
                max_incentive = incentive_right
                best_action = -1
                self.change_lane_progress=1
        
        # 检查左车道
        if current_lane < max_lane:
            safe_left, incentive_left = self.evaluate_lane_change(
                current_lane + 1, cars, lane_width, b_safe, p)
            if safe_left and incentive_left > max_incentive:
                max_incentive = incentive_left
                best_action = 1
                self.change_lane_progress=1
        
        return best_action




class BasePhysicsEnv(gym.Env):
    def __init__(self, num_cars=5, lanes=3, output_file="vehicle_trajectories.h5"):
        
        
        # --- 基础参数 ---
        self.num_cars = num_cars
        self.lanes = lanes
        self.lane_width = 4.0
        self.time_step = 0.2
        self.max_steps = 200
        
        self.output_file = output_file
        self.h5_file = h5py.File(output_file, 'a')
        self.current_episode = len(self.h5_file.keys())
        
        self.cars = []

        # --- 关键修改 1: 动作空间对齐 ---
        # 纵向加速度: [-4, 2] 分 31 个值
        self.acc_bins = np.linspace(-4, 2, 31)
        # 横向位移: [-0.5, 0.5] 分 10 个值 (对应 VQ-VAE 的 dy)
        self.lat_bins = np.linspace(-0.5, 0.5, 10)
        
        # [核心修改] 动作空间直接变为连续的物理量！
        # 纵向加速度 acc ∈ [-4, 2], 横向位移 dy ∈ [-0.5, 0.5]
        self.action_space = spaces.Box(
            low=np.array([-4.0, -0.5]), 
            high=np.array([2.0, 0.5]), 
            dtype=np.float32
        )
        
        # --- 关键修改 2: 状态空间对齐 ---
        # Shape: (num_cars, 4) -> [x, y, vx, vy]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)

    def _generate_vehicles(self):
        self.cars = []
        # 1. 生成 RL Agent (Ego/Adversary)
        # 随机放在任意车道，速度随机
        agent_lane = np.random.randint(1, self.lanes + 1)
        agent_pos = np.random.uniform(0, 100)
        agent = Car(id=0, lane=agent_lane, pos=agent_pos, 
                    lane_pos=(agent_lane-0.5)*self.lane_width, 
                    speed=np.random.uniform(20, 25), is_ego=True)
        self.cars.append(agent)
        
        # 2. 生成背景车 (避免重叠)
        for i in range(1, self.num_cars):
            valid = False
            while not valid:
                lane = np.random.randint(1, self.lanes + 1)
                pos = np.random.uniform(0, 200)
                lane_pos = (lane - 0.5) * self.lane_width
                
                # 简单碰撞检查
                valid = True
                for existing in self.cars:
                    if existing.lane == lane and abs(existing.pos - pos) < 30:
                        valid = False
                        break
                
                if valid:
                    npc = Car(id=i, lane=lane, pos=pos, lane_pos=lane_pos, 
                              speed=np.random.uniform(15, 30), is_ego=False)
                    self.cars.append(npc)
        
        # 按 ID 排序确保 state 顺序一致
        self.cars.sort(key=lambda x: x.id)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # 关键：初始化 seed
        self.current_step = 0
        self.current_episode_reward = 0
        self.current_episode += 1
        
        # 创建 HDF5 Group
        grp_name = f"episode_{self.current_episode}"
        if grp_name not in self.h5_file:
            grp = self.h5_file.create_group(grp_name)
            grp.create_dataset("trajectories", shape=(0, self.num_cars, 4), 
                               maxshape=(None, self.num_cars, 4), dtype=np.float32)
            
        self._generate_vehicles()
        return self.get_state(),{}

    def get_state(self):
        """
        [关键修复] 返回归一化的状态矩阵
        将物理数值除以一个常数，使其落入 [-1, 1] 区间，防止神经网络梯度爆炸
        """
        state = []
        agent = self.cars[0] 
        base_x = agent.pos   
        
        # 定义归一化常数 (根据物理意义估算)
        SCALE_X = 100.0      # 假设感知范围约 100米
        SCALE_Y = 12.0       # 3车道 * 4米 = 12米
        SCALE_V = 40.0       # 最大速度约 40m/s
        
        for car in self.cars:
            # 1. 计算相对物理量
            rel_x = car.pos - base_x 
            '''
            # 2. [核心] 执行归一化 (Normalization)
            # 使用 tanh 或 simple division 都可以，这里用除法更线性
            norm_x = rel_x / SCALE_X
            norm_y = car.lane_pos / SCALE_Y
            norm_vx = car.speed / SCALE_V
            norm_vy = 0.0 
            
            # 3. 截断保护 (Clipping)
            # 防止极其偶尔的极端值 (比如 rel_x = 500) 击穿网络
            norm_x = np.clip(norm_x, -1.0, 1.0)
            norm_y = np.clip(norm_y, 0.0, 1.0)
            norm_vx = np.clip(norm_vx, 0.0, 1.0)
            '''
            norm_x = rel_x 
            norm_y = car.lane_pos 
            norm_vx = car.speed 
            norm_vy = 0.0
            state.append([norm_x, norm_y, norm_vx, norm_vy])
            
        return np.array(state, dtype=np.float32).flatten()

    def step(self, action):
        self.current_step += 1
        
        # --- 1. 动作映射 --
        
        target_acc = np.clip(action[0], -4.0, 2.0)
        target_lat_disp = np.clip(action[1], -0.5, 0.5)
        
        # B. 执行阶段：更新所有车辆的物理位置
        self._log_to_hdf5()
        out_of_road = False # 记录智能体是否冲出道路
        
        for car in self.cars:
            if car.is_ego:
                # === RL Agent (智能体) ===
                # 智能体完全由 VQ-VAE 传来的 target_acc 和 target_lat_disp 控制
                car.acceleration = target_acc
                
                # 纵向更新
                car.pos += car.speed * time_step + 0.5 * target_acc * (time_step**2)
                car.speed += target_acc * time_step
                car.speed = max(0, car.speed) # 不倒车
                
                # 横向更新 (直接加上横向位移 dy)
                car.lane_pos += target_lat_disp
                
                # 更新所在车道 ID
                car.lane = car.get_lane_from_position()
                car.update_lane_offset()
                
                # 边界惩罚检测 (如果飞出 1~3 车道)
                if car.lane_pos < 0 or car.lane_pos > self.lanes * lane_width:
                    out_of_road = True
                    # 强行拉回边界防止崩溃
                    car.lane_pos = np.clip(car.lane_pos, 0, self.lanes * lane_width)
                    car.lane = car.get_lane_from_position()
                    
            else:
                # === Background Cars (背景车) ===
                # 按照你复杂 Car 类的固有逻辑更新
                car.acceleration=car.calculate_acceleration(car.find_leading_car(self.cars))
                # 对抗背景车由智能体控制，使用强化学习模型来更新纵向加速度
                car.lane_change_direction = car.decide_lane_change(self.cars)
                car.update_lane_position()            # 处理换道平滑横向移动
                car.update_position(car.acceleration) # 处理纵向 IDM 移动
                car.update_lane_offset()              # 更新横向偏移量

        # --- 3. 奖励计算 (最小 TTC & 碰撞) ---
        reward = 0
        done = False
        collision = False
        min_ttc = 100.0
        truncated = False
        agent = self.cars[0] # 假设 ID 0 是 Agent
        
        # 遍历计算 Agent 与其他车的 TTC 和 碰撞
        for npc in self.cars:
            if npc.id == agent.id: continue
            
            # 碰撞检测 (AABB)
            if self._check_collision(agent, npc):
                collision = True
                break
            
            # TTC 计算
            dist_x = npc.pos - agent.pos
            rel_v = agent.speed - npc.speed
            
            # 仅在同车道或横向距离很近时计算风险
            if abs(agent.lane_pos - npc.lane_pos) < 2.2:
                # Agent 追尾前车
                if dist_x > 0 and rel_v > 0:
                    #ttc = (dist_x - 5) / rel_v
                    #min_ttc = min(min_ttc, ttc)
                    pass
                # 后车追尾 Agent (被动风险)
                elif dist_x < 0 and rel_v < 0:
                    ttc = (abs(dist_x) - 5) / abs(rel_v)
                    min_ttc = min(min_ttc, ttc)
        if min_ttc < 1e-5: min_ttc = 1e-5
        # --- 奖励逻辑 ---
        if collision: 
            # 发生碰撞，回合结束
            # 如果是生成高危场景（Attack），碰撞给大奖
            
            done = True
            if min_ttc<10:
                reward = 100.0 
        else:
            # 未碰撞，奖励与 TTC 负相关 (TTC 越小越危险，奖励越高)
            # 使用 exp(-TTC) 将 TTC 映射到 (0, 1]
            if min_ttc<3:
                reward = np.exp(-min_ttc / 3.0) *100
                done = True
            else:
                reward = -0.01 # 稍微惩罚无风险的游荡
                if self.current_step >= self.max_steps:
                    truncated = True                

            
            # 步数惩罚/最大步数截断
        reward = np.clip(reward, -10.0, 100.0)    

        self.current_episode_reward += reward
        
        info = {
            'collision': collision,
            'min_ttc': min_ttc,
            'episode_reward': self.current_episode_reward
        }
        terminated = done

        if terminated or truncated:
            grp_name = f"episode_{self.current_episode}"
            if grp_name in self.h5_file:
                grp = self.h5_file[grp_name]
                grp.attrs['episode_reward'] = float(self.current_episode_reward)
                grp.attrs['episode_length'] = int(self.current_step)
                grp.attrs['collision'] = bool(collision)
                    
        return self.get_state(), reward, terminated, truncated, info


    def _find_leader(self, car):
        """寻找当前车道最近前车"""
        min_dist = float('inf')
        leader = None
        for other in self.cars:
            if other.id == car.id: continue
            # 判定同车道
            if abs(other.lane_pos - car.lane_pos) < 2.0:
                dist = other.pos - car.pos
                if 0 < dist < min_dist:
                    min_dist = dist
                    leader = other
        return leader

    def _check_collision(self, car1, car2):
        # 简单矩形碰撞
        dx = abs(car1.pos - car2.pos)
        dy = abs(car1.lane_pos - car2.lane_pos)
        return dx < (car1.length/2 + car2.length/2) and dy < (car1.width/2 + car2.width/2)

    def _log_to_hdf5(self):
        data = np.zeros((1, self.num_cars, 4), dtype=np.float32)
        for i, car in enumerate(self.cars):
            data[0, i, :] = [car.pos, car.lane_pos, car.speed, 0]
            
        dset = self.h5_file[f"episode_{self.current_episode}/trajectories"]
        dset.resize(dset.shape[0]+1, axis=0)
        dset[-1:] = data



from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv



time.sleep(1)  # 确保文件名唯一
timestamp = time.strftime("%Y%m%d_%H%M%S")  # 修正格式字符串
record_path = f"./ppo_logs_{timestamp}/"
if not os.path.exists(record_path):
    os.makedirs(record_path)  # 自动创建文件夹

filename = record_path + f"vae-ppo_vehicle_trajectories_{timestamp}.h5"
# --- 1. 创建环境 ---
# 确保你的 TrafficEnv 已经使用了上面的 gymnasium 修改版
raw_env = BasePhysicsEnv(output_file=filename)
# 使用修复后的 Adapter
#raw_env = HierarchicalAdapter(raw_env, 'hierarchical_checkpoints/low_level.pth', CONFIG)
# 放入 VecEnv
vec_env = DummyVecEnv([lambda: raw_env])

# 创建PPO2D模型
model = PPO(
    policy="MlpPolicy",
    env=vec_env,
    
    verbose=1,
    device='cpu'
)

model.learn(total_timesteps=200000)
model.save(record_path +"ppo_200k")