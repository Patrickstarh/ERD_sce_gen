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
import copy
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
    def __init__(self, id, lane, pos, lane_pos, speed, length=5, width=2, lanes=3, is_ego=0, is_adversary=0):
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
        
        self.is_ego = is_ego  # 是否为自车（被测试系统：观测视角 + 感知噪声注入对象）
        self.is_adversary = is_adversary  # 是否为被 RL 控制的对抗背景车
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

        # 被 RL 控制的对抗背景车 id（背景车 id 从 1 开始；其余背景车与自车一样用 IDM + MOBIL）
        self.adversary_id = 1

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

        # ===== 感知噪声模型：距离依赖的二次多项式高斯分布 =====
        # mu(d)    = a0 + a1*d + a2*d^2
        # sigma(d) = b0 + b1*d + b2*d^2

        self.perception_noise_coeffs = {
            'x': {
                'mu':    (-0.00691,  -0.00013,  -4.64218e-07),
                'sigma': ( 0.03711,   0.00511,  -5.68800e-05),
            },
            'y': {
                'mu':    (-0.00571,   0.00039,  -4.90203e-08),
                'sigma': ( 0.06703,   0.00593,  -7.43626e-05),
            },
            'vx': {
                'mu':    (-0.00369,   0.00054,  -6.30793e-06),
                'sigma': ( 0.06579,   0.00693,  -7.20104e-05),
            },
            'vy': {
                'mu':    ( 0.00864,  -0.00323,   6.23654e-05),
                'sigma': ( 0.31881,   0.00069,   5.11978e-06),
            },
        }

        self.enable_perception_noise = True  # 消融实验时可设为 False


    def _generate_vehicles(self):
        self.cars = []
        # 1. 生成自车（被测试系统，正常驾驶，由 IDM + MOBIL 控制，不做 RL）
        agent_lane = np.random.randint(1, self.lanes + 1)
        agent_pos = np.random.uniform(0, 100)
        agent = Car(id=0, lane=agent_lane, pos=agent_pos,
                    lane_pos=(agent_lane-0.5)*self.lane_width,
                    speed=np.random.uniform(20, 25), is_ego=True, is_adversary=False)
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
                              speed=np.random.uniform(15, 30), is_ego=False,
                              is_adversary=(i == self.adversary_id))
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
            grp.create_dataset("perception_data",
                            shape=(0, self.num_cars, 13),
                            maxshape=(None, self.num_cars, 13),
                            dtype=np.float32)

        self.last_perceived_states = np.zeros((self.num_cars, 13), dtype=np.float32)

        self._generate_vehicles()
        self._update_perception()  # 采样并缓存自车的带噪感知（供首步决策与日志）
        init_state = self.get_true_state()  # RL（对抗车）观测使用真实状态
        return init_state, {}

    def _update_perception(self):
        """
        采样一次自车对周围车辆的带噪感知（基于当前真实状态），
        并把结果存到各车的 perceived_*/delta_*/dist_to_ego 字段，
        同时构建 self.last_perceived_states（13 列：感知值/真实值/误差/距离）供日志使用。
        只有自车有感知受限（对非自车加噪声）；自车对自身、以及背景车之间均为真实感知。
        """
        ego = self.cars[0]
        perceived_log = []

        for car in self.cars:
            if car.is_ego:
                car.dist_to_ego = 0.0
                car.delta_x = car.delta_y = car.delta_vx = car.delta_vy = 0.0
                car.perceived_pos = car.pos
                car.perceived_lane_pos = car.lane_pos
                car.perceived_speed = car.speed
                perceived_log.append([
                    car.pos, car.lane_pos, car.speed, 0.0,   # 感知值
                    car.pos, car.lane_pos, car.speed, 0.0,   # 真实值
                    0.0, 0.0, 0.0, 0.0,                      # 误差 delta
                    0.0                                       # 距离
                ])
            else:
                dist = max(0.0, np.sqrt((car.pos - ego.pos) ** 2 +
                                        (car.lane_pos - ego.lane_pos) ** 2))
                car.dist_to_ego = dist
                if self.enable_perception_noise:
                    dx, dy, dvx, dvy = self._sample_perception_noise(dist)
                else:
                    dx = dy = dvx = dvy = 0.0
                car.delta_x, car.delta_y, car.delta_vx, car.delta_vy = dx, dy, dvx, dvy
                car.perceived_pos = car.pos + dx
                car.perceived_lane_pos = car.lane_pos + dy
                car.perceived_speed = max(0.0, car.speed + dvx)
                perceived_log.append([
                    car.perceived_pos, car.perceived_lane_pos, car.perceived_speed, dvy,  # 感知值
                    car.pos,           car.lane_pos,           car.speed,           0.0,  # 真实值
                    dx,                dy,                     dvx,                 dvy,  # 误差
                    dist                                                                  # 距离
                ])

        self.last_perceived_states = np.array(perceived_log, dtype=np.float32)

    def _ego_perceived_cars(self):
        """
        返回自车眼中的车辆列表：自车为真实自身，其余车辆用带噪感知的位置/速度。
        供自车的 IDM / MOBIL 决策使用（浅拷贝，不修改真实车辆状态）。
        """
        perceived = []
        for car in self.cars:
            if car.is_ego:
                perceived.append(car)
            else:
                c = copy.copy(car)
                c.pos = car.perceived_pos
                c.lane_pos = car.perceived_lane_pos
                c.speed = car.perceived_speed
                perceived.append(c)
        return perceived

    def get_true_state(self):
        """
        返回所有车辆的真实状态（无感知噪声），供对抗背景车的 RL 观测使用。
        对抗车是真实物理车辆，没有感知受限问题；感知误差是自车的属性，
        已通过奖励 r_perception 体现。
        格式：[x, y, vx, vy] * num_cars（以自车为原点）。
        """
        base_x = self.cars[0].pos
        state = []
        for car in self.cars:
            state.append([car.pos - base_x, car.lane_pos, car.speed, 0.0])
        return np.array(state, dtype=np.float32).flatten()

    def step(self, action):
        self.current_step += 1

        target_acc = np.clip(action[0], -4.0, 2.0)
        target_lat_disp = np.clip(action[1], -0.5, 0.5)

        out_of_road = False

        # === 0. 采样一次自车的带噪感知（本步决策与日志共用同一次噪声）===
        self._update_perception()

        # === 1. 物理更新 ===
        # 对抗背景车由 RL 控制（制造危险）；自车基于带噪感知做 IDM + MOBIL；
        # 其他背景车基于真实感知做 IDM + MOBIL
        for car in self.cars:
            if car.is_adversary:
                car.acceleration = target_acc
                car.pos += car.speed * time_step + 0.5 * target_acc * (time_step**2)
                car.speed += target_acc * time_step
                car.speed = max(0, car.speed)
                car.lane_pos += target_lat_disp
                car.lane = car.get_lane_from_position()
                car.update_lane_offset()
                if car.lane_pos < 0 or car.lane_pos > self.lanes * lane_width:
                    out_of_road = True
                    car.lane_pos = np.clip(car.lane_pos, 0, self.lanes * lane_width)
                    car.lane = car.get_lane_from_position()
            elif car.is_ego:
                perceived_cars = self._ego_perceived_cars()
                car.acceleration = car.calculate_acceleration(car.find_leading_car(perceived_cars))
                car.lane_change_direction = car.decide_lane_change(perceived_cars)
                car.update_lane_position()
                car.update_position(car.acceleration)
                car.update_lane_offset()
            else:
                car.acceleration = car.calculate_acceleration(car.find_leading_car(self.cars))
                car.lane_change_direction = car.decide_lane_change(self.cars)
                car.update_lane_position()
                car.update_position(car.acceleration)
                car.update_lane_offset()

        # === 2. 生成当前步观测（RL 对抗车用真实状态）===
        current_state = self.get_true_state()  # RL（对抗车）观测使用真实状态

        # === 3. 记录数据 ===
        self._log_to_hdf5()

        # === 3.5 越界检测：直接终止并惩罚 ===
        if out_of_road:
            reward = -5.0
            self.current_episode_reward += reward
            info = {
                'collision': False,
                'min_ttc': 100.0,
                'r_safety': 0.0,
                'r_perception': 0.0,
                'perception_delta': 0.0,
                'episode_reward': self.current_episode_reward
            }
            grp_name = f"episode_{self.current_episode}"
            if grp_name in self.h5_file:
                grp = self.h5_file[grp_name]
                grp.attrs['episode_reward'] = float(self.current_episode_reward)
                grp.attrs['episode_length'] = int(self.current_step)
                grp.attrs['collision'] = False
            return current_state, reward, True, False, info

        # === 4. 碰撞检测 & 真实TTC计算 ===
        collision = False
        min_ttc = 100.0
        agent = self.cars[0]

        for npc in self.cars:
            if npc.id == agent.id:
                continue
            if self._check_collision(agent, npc):
                collision = True
                break
            dist_x = npc.pos - agent.pos
            rel_v = agent.speed - npc.speed
            if abs(agent.lane_pos - npc.lane_pos) < 2.2:
                if dist_x > 0 and rel_v > 0:
                    # 自车追上前方慢车（前向追尾风险）
                    ttc = max((dist_x - agent.length), 0.0) / rel_v
                    min_ttc = min(min_ttc, ttc)
                elif dist_x < 0 and rel_v < 0:
                    # 后车追上自车
                    ttc = max((abs(dist_x) - agent.length), 0.0) / abs(rel_v)
                    min_ttc = min(min_ttc, ttc)

        if min_ttc < 1e-5:
            min_ttc = 1e-5

        # === 5. 安全临界场景奖励 r_safety（论文公式16）===
        TTC_LOW      = 1.0
        TTC_HIGH     = 4.0
        alpha_safety = 5.0
        beta_safety  = 1.0

        done = False
        truncated = False

        if collision or min_ttc < TTC_LOW:
            r_safety = beta_safety
            done = True
        elif TTC_LOW <= min_ttc <= TTC_HIGH:
            r_safety = alpha_safety * (TTC_HIGH - min_ttc) / (TTC_HIGH - TTC_LOW)
        else:
            r_safety = 0.0
            if self.current_step >= self.max_steps:
                truncated = True

        # === 6. 感知误差安全风险奖励 r_perception（论文公式17）===
        alpha_perc = 10.0
        beta_perc  = 20.0

        r_perception, Delta = self._compute_perception_risk_reward(
            n_samples=10,
            n_steps=10,
            alpha_perc=alpha_perc,
            beta_perc=beta_perc
        )

        # === 7. 综合奖励（论文公式15）===
        reward = r_safety + r_perception
        reward = np.clip(reward, -10.0, 100.0)

        self.current_episode_reward += reward

        info = {
            'collision': collision,
            'min_ttc': min_ttc,
            'r_safety': r_safety,
            'r_perception': r_perception,
            'perception_delta': Delta,
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

        return current_state, reward, terminated, truncated, info


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
        # --- 原有：记录真实轨迹 ---
        true_data = np.zeros((1, self.num_cars, 4), dtype=np.float32)
        for i, car in enumerate(self.cars):
            true_data[0, i, :] = [car.pos, car.lane_pos, car.speed, 0]
        
        dset = self.h5_file[f"episode_{self.current_episode}/trajectories"]
        dset.resize(dset.shape[0] + 1, axis=0)
        dset[-1:] = true_data

        # --- 新增：记录感知数据（含误差和距离）---
        perc_data = self.last_perceived_states[np.newaxis, :, :]  # (1, num_cars, 13)
        
        dset_perc = self.h5_file[f"episode_{self.current_episode}/perception_data"]
        dset_perc.resize(dset_perc.shape[0] + 1, axis=0)
        dset_perc[-1:] = perc_data

    def _poly2(self, coeffs, d):
        """计算二次多项式: a0 + a1*d + a2*d^2，并裁剪sigma为非负"""
        a0, a1, a2 = coeffs
        return a0 + a1 * d + a2 * (d ** 2)

    def _sample_perception_noise(self, d):
        """
        给定与背景车的距离d（米），采样四维感知噪声。
        返回: (delta_x, delta_y, delta_vx, delta_vy)
        """
        d = max(0.0, d)  # 距离非负保护
        noises = {}
        for var, coeffs in self.perception_noise_coeffs.items():
            mu    = self._poly2(coeffs['mu'],    d)
            sigma = self._poly2(coeffs['sigma'], d)
            sigma = max(sigma, 1e-6)  # sigma必须为正
            noises[var] = np.random.normal(mu, sigma)
        return noises['x'], noises['y'], noises['vx'], noises['vy']

    def _simulate_future_ttc(self, observed_states, n_steps=10):
        """
        基于给定的观测状态（真实或带噪），使用代理模型前向仿真n步，返回最小TTC。
        
        observed_states: list of dict，每辆车的观测状态
            格式: [{'pos': x, 'lane_pos': y, 'speed': vx, 'id': i, 'is_ego': bool}, ...]
        n_steps: 前向仿真步数
        返回: 该条轨迹的最小TTC（float）
        """
        # 深拷贝状态，避免修改真实环境
        sim_cars = []
        for s in observed_states:
            c = Car(
                id=s['id'],
                lane=int(s['lane_pos'] // self.lane_width) + 1,
                pos=s['pos'],
                lane_pos=s['lane_pos'],
                speed=max(0.0, s['speed']),
                is_ego=s['is_ego']
            )
            sim_cars.append(c)

        min_ttc = 100.0

        for _ in range(n_steps):
            # 更新每辆车（全部用IDM代理模型，包括ego）
            for car in sim_cars:
                leading = car.find_leading_car(sim_cars)
                acc = car.calculate_acceleration(leading)
                car.pos += car.speed * time_step + 0.5 * acc * (time_step ** 2)
                car.speed = max(0.0, car.speed + acc * time_step)

            # 计算ego与其他车的TTC
            ego = sim_cars[0]
            for npc in sim_cars[1:]:
                dist_x = npc.pos - ego.pos
                rel_v = ego.speed - npc.speed
                if abs(ego.lane_pos - npc.lane_pos) < 2.2:
                    if dist_x > 0 and rel_v > 0:
                        ttc = max((dist_x - ego.length), 0.0) / rel_v
                        min_ttc = min(min_ttc, ttc)
                    elif dist_x < 0 and rel_v < 0:
                        ttc = max((abs(dist_x) - ego.length), 0.0) / abs(rel_v)
                        min_ttc = min(min_ttc, ttc)

        return min_ttc

    def _compute_perception_risk_reward(self, n_samples=10, n_steps=10,
                                        alpha_perc=1.0, beta_perc=2.0):
        """
        计算感知误差安全风险奖励（论文公式17）。
        
        核心逻辑：
        1. 用真实观测前向仿真 n_samples 条轨迹 → 真实风险分布 E[R_true]
        2. 用带噪观测前向仿真 n_samples 条轨迹 → 感知风险分布 E[R_obs]  
        3. 计算期望风险差 Delta = E[R_obs] - E[R_true]
        4. 奖励 = alpha * |Delta| + beta * max(0, -Delta)
        
        返回: (perception_reward, Delta)
        """
        agent = self.cars[0]

        # --- 构建真实观测状态 ---
        true_states = []
        for car in self.cars:
            true_states.append({
                'id': car.id,
                'pos': car.pos,
                'lane_pos': car.lane_pos,
                'speed': car.speed,
                'is_ego': car.is_ego
            })

        # --- 多次采样带噪观测状态 ---
        def build_noisy_states():
            noisy = []
            for car in self.cars:
                if car.is_ego:
                    noisy.append({
                        'id': car.id,
                        'pos': car.pos,
                        'lane_pos': car.lane_pos,
                        'speed': car.speed,
                        'is_ego': True
                    })
                else:
                    dist = np.sqrt((car.pos - agent.pos) ** 2 +
                                (car.lane_pos - agent.lane_pos) ** 2)
                    if self.enable_perception_noise:
                        dx, dy, dvx, _ = self._sample_perception_noise(dist)
                    else:
                        dx, dy, dvx = 0.0, 0.0, 0.0
                    noisy.append({
                        'id': car.id,
                        'pos': car.pos + dx,
                        'lane_pos': car.lane_pos + dy,
                        'speed': max(0.0, car.speed + dvx),
                        'is_ego': False
                    })
            return noisy

        # --- 采样风险分布 ---
        # 注意：TTC越小风险越高，这里用 1/TTC 作为风险度量，TTC=100时风险≈0
        def ttc_to_risk(ttc):
            return 1.0 / max(ttc, 0.1)   # 风险值，TTC越小风险越大

        # 真实风险只需仿真一次（确定性，无随机）
        true_ttc = self._simulate_future_ttc(true_states, n_steps)
        E_true = ttc_to_risk(true_ttc)

        # 带噪风险 Monte Carlo 采样 n_samples 次
        obs_risks = []
        for _ in range(n_samples):
            noisy_states = build_noisy_states()
            obs_ttc = self._simulate_future_ttc(noisy_states, n_steps)
            obs_risks.append(ttc_to_risk(obs_ttc))
        E_obs = np.mean(obs_risks)

        Delta = E_obs - E_true
        perception_reward = alpha_perc * abs(Delta) + beta_perc * max(0.0, -Delta)
        return perception_reward, Delta



import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


def parse_args():
    parser = argparse.ArgumentParser(description="GPU 版 RL 场景生成训练脚本")
    parser.add_argument('--gpu', type=int, default=0, help='使用哪张 GPU（多卡各自独立跑）')
    parser.add_argument('--timesteps', type=int, default=200000, help='总训练步数')
    parser.add_argument('--seed', type=int, default=0, help='随机种子（多卡时每张卡用不同 seed）')
    return parser.parse_args()


args = parse_args()
device = f'cuda:{args.gpu}'



time.sleep(1)  # 确保文件名唯一
timestamp = time.strftime("%Y%m%d_%H%M%S")  # 修正格式字符串
record_path = f"./ppo_logs_gpu{args.gpu}_{timestamp}/"
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
    device=device,
    seed=args.seed,
    tensorboard_log=record_path + "tensorboard"
)

model.learn(total_timesteps=args.timesteps)
model.save(record_path + f"ppo_{args.timesteps // 1000}k")