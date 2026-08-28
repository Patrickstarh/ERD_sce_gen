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

# Simulation parameters
lane_width = 4.0  # Lane width (meters)
time_step = 0.2  # Simulation time per step (seconds)
lanes = 3  # Three lanes
total_time = 30  # Total simulation duration (seconds)
collision_threshold = 2.0  # Collision threshold (meters)
num_cars = 5  # Number of vehicles
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
        self.lane_offset = 0  # Lateral offset (from lane center)
        self.length = length  # Vehicle length
        self.width = width  # Vehicle width
        self.change_lane_progress = 0  # Lane-change progress
        self.lane_change_direction = ''
        self.lanes = lanes

        # Fix 2: use the passed-in lane_pos directly instead of recomputing it
        self.lane_pos = lane_pos

        self.is_ego = is_ego  # Whether this is the ego vehicle (system under test: observation viewpoint + perception-noise injection target)
        self.is_adversary = is_adversary  # Whether this is the RL-controlled adversarial background vehicle
        self.acceleration = 0  # Default acceleration is 0
        self.leading_distance = -1  # Default leading distance is -1
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
        Update the longitudinal position using the IDM model.

        :param acceleration: longitudinal acceleration of the vehicle
        """
        self.pos += self.speed*time_step+0.5*acceleration*time_step**2
        self.speed += acceleration*time_step
        if self.speed<0:
            self.speed=0


    def update_lane_position(self):
        """
        Smooth the lateral displacement during a lane change.
        """
        if self.change_lane_progress > 0 and self.change_lane_progress <= 5:
            # Lateral displacement of the lane change; left is positive, right is negative
            delta_offset_list =[i*lane_width for i in [0.1, 0.2, 0.4, 0.2, 0.1]]  # Lateral displacement amount of the lane change
            delta_offset=delta_offset_list[self.change_lane_progress - 1]
            #print('0')
            if self.lane_change_direction == "right":
                self.lane_offset -= delta_offset
                self.lane_pos-=delta_offset
                self.lane=self.get_lane_from_position()
                #print('1')
            else:  # Left lane change
                self.lane_offset += delta_offset
                self.lane_pos+=delta_offset
                self.lane=self.get_lane_from_position()
                #print('2')
            self.change_lane_progress += 1  # Increment lane-change progress
        # Reset progress after the lane change completes
        if self.change_lane_progress > 5:
            self.change_lane_progress = 0  # Reset progress to zero after completion

        # When lane-change progress is negative, return to the nearest lane center as soon as possible
        if self.change_lane_progress < 0:
            target_lane_center = (self.lane - 0.5) * lane_width  # Center position of the current lane
            lateral_diff = target_lane_center - self.lane_pos  # Lateral offset
            max_lateral_move = lane_width / 4  # Maximum lateral move is one quarter of the lane width

            # Adjust lateral position according to the maximum displacement
            if abs(lateral_diff) <= max_lateral_move:
                self.lane_pos = target_lane_center  # Move directly to the lane center
                self.change_lane_progress = 0  # Reset lane-change progress
            else:
                self.lane_pos += max_lateral_move if lateral_diff > 0 else -max_lateral_move
                self.lane = self.get_lane_from_position()  # Update lane index

    def update_lane_offset(self):
        """
        Update the offset of the vehicle relative to the nearest lane centerline.
        """
        # Compute the position of the nearest lane centerline
        target_lane_center = (self.lane - 0.5) * lane_width
        # Compute the offset of the vehicle relative to the lane centerline
        self.lane_offset = self.lane_pos - target_lane_center

    def get_lane_from_position(self):
        lane_id = int(self.lane_pos // lane_width)+1
        return lane_id

    def get_lateral_range(self, lane_width=4):
        """Compute the vehicle's current lateral coverage range [start, end]."""
        actual_lateral = self.lane_pos
        return (actual_lateral - self.width/2, actual_lateral + self.width/2)

    def is_overlapping(self, other_car, lane_width=4):
        """Check whether the lateral range overlaps with another vehicle."""
        s_start, s_end = self.get_lateral_range(lane_width)
        o_start, o_end = other_car.get_lateral_range(lane_width)
        return (s_start < o_end) and (s_end > o_start)

    def find_leading_car(self, cars):
        """Find the nearest leading vehicle within the current lateral range."""
        leading = None
        for car in cars:
            if car.id == self.id or not self.is_overlapping(car) or car.pos <= self.pos:
                continue
            if leading is None or car.pos < leading.pos:
                leading = car
        self.leading_distance = leading.pos - self.pos if leading else -1
        return leading

    # --- MOBIL lane-change evaluation ---
    def evaluate_lane_change(self, target_lane, cars, lane_width=4, b_safe=4.0, p=0.1):
        """Evaluate the safety and incentive of changing to the target lane.

        Returns: (safe, incentive)
        """
        # 1. Compute the lateral coverage range after the lane change
        target_center = (target_lane - 0.5) * lane_width
        new_lateral_start = target_center - self.width/2-lane_width/4
        new_lateral_end = target_center + self.width/2+lane_width/4

        # 2. Find the nearest leading and following vehicles in the target lane
        leading, following = None, None
        for car in cars:
            if car.id == self.id:
                continue
            # Check whether the vehicle is within the target lane's lateral range
            c_start, c_end = car.get_lateral_range(lane_width)
            if (new_lateral_start < c_end) and (new_lateral_end > c_start):
                if car.pos > self.pos:  # Candidate leading vehicle
                    if leading is None or car.pos < leading.pos:
                        leading = car
                else:  # Candidate following vehicle
                    if following is None or car.pos > following.pos:
                        following = car

        # 3. Safety check: can the following vehicle in the target lane brake safely?
        safe = True
        if following:
            # Compute the following vehicle's acceleration after the lane change (assuming its new leader is self)
            a_back_new = following.calculate_acceleration(leading_car=self)
            safe = (a_back_new >= -b_safe)

        # 4. Compute the incentive (core MOBIL formula)
        # Current own acceleration (original lane)
        current_leader = self.find_leading_car(cars)
        a_current = self.calculate_acceleration(current_leader)

        # Own acceleration after the lane change (target lane)
        a_new = self.calculate_acceleration(leading)
        #if leading else self.get_free_acceleration()

        # Acceleration of the following vehicle in the original lane (if any)
        current_follower = None
        for car in cars:
            if car.id == self.id or not self.is_overlapping(car) or car.pos >= self.pos:
                continue
            if current_follower is None or car.pos > current_follower.pos:
                current_follower = car
        a_old_back = current_follower.calculate_acceleration(self) if current_follower else 0

        # New acceleration of the following vehicle in the target lane (if any)
        a_new_back = following.calculate_acceleration(self) if following else 0

        # MOBIL incentive: a_new - a_current + p*(a_new_back - a_old_back) > threshold
        incentive = (a_new - a_current) + p * (a_new_back - a_old_back)

        return safe, incentive

    # --- Lane-change decision function ---
    def decide_lane_change(self, cars, max_lane=3, lane_width=4, b_safe=4.0, p=0.1, threshold=1/num_cars):
        """Core MOBIL lane-change decision logic.

        Returns: 'stay', 'left', or 'right'
        """
        if self.change_lane_progress > 0 or self.change_lane_progress < 0:
            return self.lane_change_direction  # Do not re-decide while a lane change is in progress

        current_lane = self.lane
        best_action = 'stay'

        max_incentive = threshold  # Incentive must exceed the threshold


        # Check the right lane
        if current_lane > 1:
            safe_right, incentive_right = self.evaluate_lane_change(
                current_lane - 1, cars, lane_width, b_safe, p
            )
            if safe_right and incentive_right > max_incentive+0.1:
                max_incentive = incentive_right
                best_action = 'right'
                self.change_lane_progress=1

        # Check the left lane
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
        """Core MOBIL lane-change decision logic.

        Returns: 'stay', 'left', or 'right'
        """
        if self.change_lane_progress > 4:
            self.lane_change_progress=0

        if self.change_lane_progress > 0:
            return lanechange[self.lane_change_direction]  # Do not re-decide while a lane change is in progress

        current_lane = self.lane
        best_action = 0
        max_incentive = threshold  # Incentive must exceed the threshold

        # Check the right lane
        if current_lane > 1:
            safe_right, incentive_right = self.evaluate_lane_change(
                current_lane - 1, cars, lane_width, b_safe, p
            )
            if safe_right and incentive_right > max_incentive+0.1:
                max_incentive = incentive_right
                best_action = -1
                self.change_lane_progress=1

        # Check the left lane
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


        # --- Basic parameters ---
        self.num_cars = num_cars
        self.lanes = lanes
        self.lane_width = 4.0
        self.time_step = 0.2
        self.max_steps = 200

        self.output_file = output_file
        self.h5_file = h5py.File(output_file, 'a')
        self.current_episode = len(self.h5_file.keys())

        self.cars = []

        # Id of the RL-controlled adversarial background vehicle (background ids start at 1; other background vehicles use IDM + MOBIL like the ego)
        self.adversary_id = 1

        # --- Key change 1: action-space alignment ---
        # Longitudinal acceleration: [-4, 2] split into 31 values
        self.acc_bins = np.linspace(-4, 2, 31)
        # Lateral displacement: [-0.5, 0.5] split into 10 values (corresponds to VQ-VAE's dy)
        self.lat_bins = np.linspace(-0.5, 0.5, 10)

        # [Core change] Action space is now continuous physical quantities!
        # Longitudinal acceleration acc in [-4, 2], lateral displacement dy in [-0.5, 0.5]
        self.action_space = spaces.Box(
            low=np.array([-4.0, -0.5]),
            high=np.array([2.0, 0.5]),
            dtype=np.float32
        )

        # --- Key change 2: state-space alignment ---
        # Shape: (num_cars, 4) -> [x, y, vx, vy]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)

        # ===== Perception-noise model: distance-dependent quadratic Gaussian =====
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

        self.enable_perception_noise = True  # Set to False for ablation studies


    def _generate_vehicles(self):
        self.cars = []
        # 1. Create the ego vehicle (system under test, normal driving, controlled by IDM + MOBIL, no RL)
        agent_lane = np.random.randint(1, self.lanes + 1)
        agent_pos = np.random.uniform(0, 100)
        agent = Car(id=0, lane=agent_lane, pos=agent_pos,
                    lane_pos=(agent_lane-0.5)*self.lane_width,
                    speed=np.random.uniform(20, 25), is_ego=True, is_adversary=False)
        self.cars.append(agent)

        # 2. Create background vehicles (avoiding overlap)
        for i in range(1, self.num_cars):
            valid = False
            while not valid:
                lane = np.random.randint(1, self.lanes + 1)
                pos = np.random.uniform(0, 200)
                lane_pos = (lane - 0.5) * self.lane_width

                # Simple collision check
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

        # Sort by id to keep state order consistent
        self.cars.sort(key=lambda x: x.id)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # Important: initialize the seed
        self.current_step = 0
        self.current_episode_reward = 0
        self.current_episode += 1

        # Create the HDF5 group
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
        self._update_perception()  # Sample and cache the ego's noisy perception (for first-step decision and logging)
        init_state = self.get_true_state()  # The RL (adversarial) observation uses the true state
        return init_state, {}

    def _update_perception(self):
        """
        Sample the ego's noisy perception of surrounding vehicles once (based on the current true state),
        store the results in each car's perceived_*/delta_*/dist_to_ego fields, and build
        self.last_perceived_states (13 columns: perceived/true/error/distance) for logging.
        Only the ego has limited perception (noise is added to non-ego vehicles); the ego's
        perception of itself and background vehicles' perception of each other are all true.
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
                    car.pos, car.lane_pos, car.speed, 0.0,   # perceived
                    car.pos, car.lane_pos, car.speed, 0.0,   # true
                    0.0, 0.0, 0.0, 0.0,                      # error delta
                    0.0                                       # distance
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
                    car.perceived_pos, car.perceived_lane_pos, car.perceived_speed, dvy,  # perceived
                    car.pos,           car.lane_pos,           car.speed,           0.0,  # true
                    dx,                dy,                     dvx,                 dvy,  # error
                    dist                                                                  # distance
                ])

        self.last_perceived_states = np.array(perceived_log, dtype=np.float32)

    def _ego_perceived_cars(self):
        """
        Return the list of vehicles as seen by the ego: the ego itself is its true self, and the
        other vehicles use their noisy perceived position/speed. Used for the ego's IDM/MOBIL
        decisions (shallow copy, does not modify the true vehicle states).
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
        Return the true state of all vehicles (no perception noise), used as the RL observation
        of the adversarial background vehicle. The adversarial vehicle is a real physical vehicle
        without perception limitation; perception error is a property of the ego and is already
        captured by the r_perception reward.
        Format: [x, y, vx, vy] * num_cars (ego as origin).
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

        # === 0. Sample the ego's noisy perception once (decision and logging share the same noise sample) ===
        self._update_perception()

        # === 1. Physics update ===
        # The adversarial background vehicle is controlled by RL (to create danger); the ego does
        # IDM + MOBIL based on noisy perception; other background vehicles do IDM + MOBIL on true perception.
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

        # === 2. Build the current observation (RL adversarial vehicle uses true state) ===
        current_state = self.get_true_state()  # The RL (adversarial) observation uses the true state

        # === 3. Log data ===
        self._log_to_hdf5()

        # === 3.5 Out-of-road detection: terminate immediately and penalize ===
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

        # === 4. Collision detection & true TTC computation ===
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
                    # Ego catches up to a slower vehicle ahead (forward rear-end risk)
                    ttc = max((dist_x - agent.length), 0.0) / rel_v
                    min_ttc = min(min_ttc, ttc)
                elif dist_x < 0 and rel_v < 0:
                    # A following vehicle catches up to the ego
                    ttc = max((abs(dist_x) - agent.length), 0.0) / abs(rel_v)
                    min_ttc = min(min_ttc, ttc)

        if min_ttc < 1e-5:
            min_ttc = 1e-5

        # === 5. Safety-critical reward r_safety (paper Eq. 16) ===
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

        # === 6. Perception-error safety risk reward r_perception (paper Eq. 17) ===
        alpha_perc = 10.0
        beta_perc  = 20.0

        r_perception, Delta = self._compute_perception_risk_reward(
            n_samples=10,
            n_steps=10,
            alpha_perc=alpha_perc,
            beta_perc=beta_perc
        )

        # === 7. Total reward (paper Eq. 15) ===
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
        """Find the nearest leading vehicle in the current lane."""
        min_dist = float('inf')
        leader = None
        for other in self.cars:
            if other.id == car.id: continue
            # Check same-lane
            if abs(other.lane_pos - car.lane_pos) < 2.0:
                dist = other.pos - car.pos
                if 0 < dist < min_dist:
                    min_dist = dist
                    leader = other
        return leader

    def _check_collision(self, car1, car2):
        # Simple axis-aligned rectangle collision
        dx = abs(car1.pos - car2.pos)
        dy = abs(car1.lane_pos - car2.lane_pos)
        return dx < (car1.length/2 + car2.length/2) and dy < (car1.width/2 + car2.width/2)

    def _log_to_hdf5(self):
        # --- Existing: log the true trajectory ---
        true_data = np.zeros((1, self.num_cars, 4), dtype=np.float32)
        for i, car in enumerate(self.cars):
            true_data[0, i, :] = [car.pos, car.lane_pos, car.speed, 0]

        dset = self.h5_file[f"episode_{self.current_episode}/trajectories"]
        dset.resize(dset.shape[0] + 1, axis=0)
        dset[-1:] = true_data

        # --- New: log perception data (including error and distance) ---
        perc_data = self.last_perceived_states[np.newaxis, :, :]  # (1, num_cars, 13)

        dset_perc = self.h5_file[f"episode_{self.current_episode}/perception_data"]
        dset_perc.resize(dset_perc.shape[0] + 1, axis=0)
        dset_perc[-1:] = perc_data

    def _poly2(self, coeffs, d):
        """Evaluate the quadratic polynomial a0 + a1*d + a2*d^2 and clip sigma to be non-negative."""
        a0, a1, a2 = coeffs
        return a0 + a1 * d + a2 * (d ** 2)

    def _sample_perception_noise(self, d):
        """
        Sample 4D perception noise given the distance d (meters) to a background vehicle.

        Returns: (delta_x, delta_y, delta_vx, delta_vy)
        """
        d = max(0.0, d)  # Guard against negative distance
        noises = {}
        for var, coeffs in self.perception_noise_coeffs.items():
            mu    = self._poly2(coeffs['mu'],    d)
            sigma = self._poly2(coeffs['sigma'], d)
            sigma = max(sigma, 1e-6)  # sigma must be positive
            noises[var] = np.random.normal(mu, sigma)
        return noises['x'], noises['y'], noises['vx'], noises['vy']

    def _simulate_future_ttc(self, observed_states, n_steps=10):
        """
        Forward-simulate n steps with a surrogate model from the given observed states (true or noisy)
        and return the minimum TTC.

        observed_states: list of dict, the observed state of each vehicle
            format: [{'pos': x, 'lane_pos': y, 'speed': vx, 'id': i, 'is_ego': bool}, ...]
        n_steps: number of forward simulation steps
        Returns: minimum TTC of the trajectory (float)
        """
        # Build fresh Car objects to avoid mutating the real environment
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
            # Update every vehicle (all use the IDM surrogate model, including ego)
            for car in sim_cars:
                leading = car.find_leading_car(sim_cars)
                acc = car.calculate_acceleration(leading)
                car.pos += car.speed * time_step + 0.5 * acc * (time_step ** 2)
                car.speed = max(0.0, car.speed + acc * time_step)

            # Compute TTC between ego and other vehicles
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
        Compute the perception-error safety risk reward (paper Eq. 17).

        Core logic:
        1. Forward-simulate with true observations -> true risk distribution E[R_true]
        2. Forward-simulate with noisy observations (n_samples) -> perceived risk distribution E[R_obs]
        3. Compute the expected risk difference Delta = E[R_obs] - E[R_true]
        4. reward = alpha * |Delta| + beta * max(0, -Delta)

        Returns: (perception_reward, Delta)
        """
        agent = self.cars[0]

        # --- Build true observed states ---
        true_states = []
        for car in self.cars:
            true_states.append({
                'id': car.id,
                'pos': car.pos,
                'lane_pos': car.lane_pos,
                'speed': car.speed,
                'is_ego': car.is_ego
            })

        # --- Sample noisy observed states multiple times ---
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

        # --- Sample the risk distribution ---
        # Note: smaller TTC means higher risk; use 1/TTC as the risk measure (risk ~ 0 when TTC = 100)
        def ttc_to_risk(ttc):
            return 1.0 / max(ttc, 0.1)   # Risk value; smaller TTC means higher risk

        # True risk only needs one simulation (deterministic, no randomness)
        true_ttc = self._simulate_future_ttc(true_states, n_steps)
        E_true = ttc_to_risk(true_ttc)

        # Monte-Carlo sample the noisy risk n_samples times
        obs_risks = []
        for _ in range(n_samples):
            noisy_states = build_noisy_states()
            obs_ttc = self._simulate_future_ttc(noisy_states, n_steps)
            obs_risks.append(ttc_to_risk(obs_ttc))
        E_obs = np.mean(obs_risks)

        Delta = E_obs - E_true
        perception_reward = alpha_perc * abs(Delta) + beta_perc * max(0.0, -Delta)
        return perception_reward, Delta



from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv



time.sleep(1)  # Ensure a unique filename
timestamp = time.strftime("%Y%m%d_%H%M%S")  # Fixed format string
record_path = f"./ppo_logs_{timestamp}/"
if not os.path.exists(record_path):
    os.makedirs(record_path)  # Create the directory automatically

filename = record_path + f"vae-ppo_vehicle_trajectories_{timestamp}.h5"
# --- 1. Create the environment ---
# Make sure your TrafficEnv uses the gymnasium-modified version above
raw_env = BasePhysicsEnv(output_file=filename)
# Use the fixed adapter
#raw_env = HierarchicalAdapter(raw_env, 'hierarchical_checkpoints/low_level.pth', CONFIG)
# Wrap in a VecEnv
vec_env = DummyVecEnv([lambda: raw_env])

# Create the PPO model
model = PPO(
    policy="MlpPolicy",
    env=vec_env,

    verbose=1,
    device='cpu',
    tensorboard_log=record_path + "tensorboard"
)

model.learn(total_timesteps=200000)
model.save(record_path +"ppo_200k")
