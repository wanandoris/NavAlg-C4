# ==========================================================
# 1. 首先导入 sys 和 os，并添加路径
# ==========================================================
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ==========================================================
# 2. 然后导入其他模块
# ==========================================================
import math
import time
import threading
from dataclasses import dataclass

import csv
from pathlib import Path
import cv2
import numpy as np
import torch

from usvlib4ros.navigation.usv_ros2_controller import Ros2Controller
from usvlib4ros.navigation.route_plan_service import RoutePlanService
from usvlib4ros.msg.global_data import GlobalData, DictToObject, Point, Constants
from usvlib4ros.msg.parameter import Parameter
from usvlib4ros.usvRosUtil import LogUtil
from usvlib4ros.user.PP0_2 import PPO, device
from usvlib4ros.user.reward import RewardConfig, compute_reward, calc_apf_heading_diff, compute_reward_breakdown, RewardBreakdown
from usvlib4ros.user.training_logger import TrainingLogger
from usvlib4ros.user.c4_project import OnlineRewardWeightUpdater, load_reward_net, save_reward_net
from usvlib4ros.user import train_reward_from_c4
from usvlib4ros.user.train_reward_from_c4 import RewardInputNormalizer, raw_step_to_network_input
# ==================== 超参数配置 ====================
N_ACTIONS = 2          # 连续动作空间
HAS_CONTINUOUS_ACTION = True  # 是否使用连续动作空间
N_STATES = 96          # 状态维度: 前方180°激光点数(约90) + speed + rotated_speed + angel_diff + distance + obstacle_min_range + obstacle_angle
MEMORY_CAPACITY = 2000
BATCH_SIZE = 128
LR_ACTOR = 0.0002
LR_CRITIC = 0.0008
GAMMA = 0.99           # 折扣因子
K_EPOCHS = 10          # PPO更新轮数
EPS_CLIP = 0.2         # PPO裁剪系数
ACTION_STD_INIT = 0.35 # 连续动作标准差初始化
MAX_EPOCH = 4000       # 最大训练轮数
MAX_STEP_PER_EPISODE = 3000   # 每轮最大步数
MAX_EPISODE_TIME = 300  # 每轮最大时间(秒)
UPDATE_INTERVAL = 512   # PPO更新间隔(步数)
CHECKPOINT_INTERVAL = 100  # 模型保存间隔(轮数)
ENABLE_DYNAMIC_REWARD_WEIGHTS = True
REWARD_WEIGHT_UPDATE_INTERVAL = 50
REWARD_WEIGHT_MIN_BUFFER_SIZE = 128
REWARD_WEIGHT_SAVE_ON_UPDATE = True
REWARD_WEIGHT_MODEL_PATH = Path(__file__).resolve().parent / "reward_weight_net_from_csv.pth"
REWARD_INPUT_NORMALIZER_PATH = Path(__file__).resolve().parent / "reward_input_normalizer.json"
REWARD_WEIGHT_LOG_PATH = Path(__file__).resolve().parent.parent / "reward_weight_output.csv"
REWARD_TASK_SCALE = 1.0
REWARD_CONTINUOUS_SCALE = 1.5
FINAL_REWARD_MIN = -10.0
FINAL_REWARD_MAX = 10.0
AUTO_TRAIN_REWARD_NET_AFTER_COLLECTION = True
EMPTY_ROUTE_FALLBACK_TIMEOUT = 5.0
FALLBACK_ROUTE_DISTANCE_M = 20.0

# ==================== 导航常量 ====================
LASER_MAX_RANGE = 5.0        # 激光雷达有效最大距离(m)
LASER_FRONT_HALF_DEG = 180   # 前方扫描扇区角度(度),仅用于碰撞检测
COLLISION_DISTANCE = 0.6     # 碰撞判定阈值(m)
ARRIVE_DISTANCE = 1.5        # 到达目标判定阈值(m)
DEFAULT_SPEED = 1.0          # 默认速度(m/s)
OBSTACLE_SLOW_RANGE = 4.0    # 进入此范围开始减速(m)
TARGET_SLOW_RANGE = 3.0      # 接近目标时减速阈值(m)
ANGULAR_VELOCITY_MAX = 200   # 最大角速度(°/s)
ACTION_TO_DEGREE_SCALE = 1   # 动作到转向角度的缩放因子
ACTION_TO_DEGREE_CONTINOUS_SCALE = 100 # 连续动作映射到转向角度缩放因子
ACTION_TO_SPEED_CONTINOUS_SCALE = 100 # 连续动作映射到速度缩放因子
CONTROL_DT = 0.1             # 控制周期(s),用于连续动作
APF_PID_KP = 0.85            # APF-PID 比例系数
APF_PID_KI = 0.02            # APF-PID 积分系数
APF_PID_KD = 0.12            # APF-PID 微分系数
APF_PID_MAX_INTEGRAL = 2.0   # APF-PID 积分限幅
PPO_MIN_BLEND = 0.35         # 混合控制中 PPO 最小占比
TARGET_ECHO_ANGLE_WINDOW_DEG = 8.0   # 屏蔽目标回波时的角度窗口(度)
TARGET_ECHO_RANGE_MARGIN = 1.0       # 屏蔽目标回波时的距离容差(m)
TARGET_ECHO_MIN_DISTANCE = 0.5       # 太近时不再做目标回波屏蔽，避免穿障
ENABLE_APF_DEBUG_VIEW = False         # 是否打开APF方向实时调试窗口
APF_DEBUG_WINDOW_NAME = "APF Heading Debug"

# ==================== 奖励权重 ====================
REWARD_ARRIVE_BONUS = 1.0       # 到达奖励，与离线 task_reward 量级对齐
REWARD_COLLISION_PENALTY = -1.0 # 碰撞惩罚，与离线 task_reward 量级对齐
REWARD_OBSTACLE_PENALTY = -0.05 # 接近障碍物惩罚，作为普通步级别的小惩罚
REWARD_NEAR_TARGET_BONUS = 0.01 # 靠近目标奖励，作为普通步级别的小奖励
REWARD_WEIGHT_DISTANCE = 0.6    # 距离奖励权重
REWARD_WEIGHT_OBSTACLE = 0.5    # 障碍物惩罚权重
REWARD_WEIGHT_HEADING = 0.3     # 航向奖励权重


@dataclass
class StepResult:
    """step函数返回值,替代原来的元组返回。"""
    next_state: np.ndarray
    reward: float
    current_distance: float
    advise_speed: float
    advise_rotate: float
    advised_heading: float
    distance: float
    degree_aship: float
    obstacle_min_range: float
    obstacle_angle: float
    breakdown: RewardBreakdown
    task_reward: float = 0.0
    dynamic_weights: tuple[float, float, float, float] | None = None
    reward_weight_update_metrics: dict[str, float] | None = None


class PPONav:
    """基于PPO算法的USV导航控制器。"""

    Instance = None

    def startService(self):
        self.navThread = threading.Thread(target=self.run)
        self.navThread.setDaemon(True)
        self.navThread.start()

    def __init__(self, ros_ctrl: Ros2Controller, global_data: GlobalData, xyzAxis: bool = True):
        self.episode_start_time = None
        self.ros_ctrl: Ros2Controller = ros_ctrl
        self.global_data: GlobalData = global_data
        self.navThread = None

        # 数据收集控制
        self.data_collection_count = 0
        self.data_collection_max = 300
        self.data_collection_enabled = True
        self.data_collection_completed = False
        self.training_logger = TrainingLogger(root_dir="Results", summary_interval=50)
        self.csv_path = Path(__file__).resolve().parent.parent / "data_collection.csv"
        self.reward_weight_log_path = REWARD_WEIGHT_LOG_PATH
        self.reward_weight_training = False
        self._init_csv_file()
        self._init_reward_weight_log_file()
        self.reward_normalizer = None
        self.reward_weight_updater = None
        self._init_dynamic_reward_weights()

        self.last_distance = None
        self.enable_apf_debug_view = ENABLE_APF_DEBUG_VIEW
        self.apf_pid_integral = 0.0
        self.apf_pid_prev_error = 0.0

        # PPO智能体
        self.ppo_agent = PPO(
            N_STATES, N_ACTIONS, LR_ACTOR, LR_CRITIC,
            GAMMA, K_EPOCHS, EPS_CLIP, HAS_CONTINUOUS_ACTION, ACTION_STD_INIT
        )
        self.next_state = None
        self.action_size = N_ACTIONS  # 动作空间大小(用于角度映射)

        # 航线相关
        self.route = None
        self.destPoint: Point = Point()
        self.destPointIndex = -1
        self.prevPoint = Point()
        self.prevPointIndex = -1

        # 训练状态
        self.max_distance = 0.0
        self.score = 0
        self.episode_reward_sum = 0.0
        self.arrive = False
        self.done = False
        self.timeout = False
        self.arrive_distance = ARRIVE_DISTANCE
        self.episode_step_count = 0

        self.routePlaneService = RoutePlanService(
            wayPointRadius=self.arrive_distance, route=self.route
        )
        self.last_laser_scan = global_data.laser_data
        self.reward_config = RewardConfig(
            target_slow_range=TARGET_SLOW_RANGE,
            angular_velocity_max=ANGULAR_VELOCITY_MAX,
            control_dt=CONTROL_DT,
            has_continuous_action=HAS_CONTINUOUS_ACTION,
            n_actions=N_ACTIONS,
            max_episode_time=MAX_EPISODE_TIME,
        )

    def _init_dynamic_reward_weights(self):
        if not ENABLE_DYNAMIC_REWARD_WEIGHTS:
            return
        if not REWARD_WEIGHT_MODEL_PATH.exists() or not REWARD_INPUT_NORMALIZER_PATH.exists():
            LogUtil.info("动态奖励权重未启用：缺少 reward_weight_net_from_csv.pth 或 reward_input_normalizer.json")
            return

        try:
            self.reward_normalizer = RewardInputNormalizer.load(str(REWARD_INPUT_NORMALIZER_PATH))
            reward_net = load_reward_net(str(REWARD_WEIGHT_MODEL_PATH), device=str(device))
            self.reward_weight_updater = OnlineRewardWeightUpdater(
                reward_net=reward_net,
                device=str(device),
                min_buffer_size=REWARD_WEIGHT_MIN_BUFFER_SIZE,
                update_interval=REWARD_WEIGHT_UPDATE_INTERVAL,
                gamma=GAMMA,
            )
            LogUtil.info(f"动态奖励权重已加载: {REWARD_WEIGHT_MODEL_PATH}")
        except Exception as exc:
            self.reward_normalizer = None
            self.reward_weight_updater = None
            LogUtil.error(f"动态奖励权重加载失败，回退固定奖励: {exc}")

    # ==================== 训练主循环 ====================

    def run(self):
        """训练与导航主循环。"""
        self.registerParameter()

        while True:
            try:
                LogUtil.info("等待训练触发...")
                while self.global_data.device_data.task_status == 0:
                    time.sleep(0.1)

                for epoch in range(MAX_EPOCH):
                    if self.global_data.device_data.task_status == 0:
                        LogUtil.info("停止训练")
                        break

                    LogUtil.info(f"第 {epoch} 轮训练开始")

                    # 复位Unity环境
                    self.ros_ctrl.reset_unity()
                    time.sleep(0.1)
                    while self.global_data.device_data.reset_status != 2:
                        time.sleep(0.1)

                    self.ros_ctrl.set_auto_work()

                    # 加载航线：如果上位机/Unity 还没有下发航线，不要直接 continue 到下一轮，
                    # 否则会出现“第 0/1/2...轮训练开始”快速刷屏但船不动。
                    self.route = self.ros_ctrl.getRoute()
                    route_wait_start = time.time()
                    while len(self.route.points) == 0:
                        if self.global_data.device_data.task_status == 0:
                            LogUtil.info("停止训练：等待航线时收到停止信号")
                            break
                        LogUtil.error("航线数据为空，等待上位机/Unity 下发航线...")
                        if (time.time() - route_wait_start) >= EMPTY_ROUTE_FALLBACK_TIMEOUT:
                            self.route = self._build_fallback_route()
                            LogUtil.info(f"未收到航线，已生成临时采集航线: {self.route}")
                            break
                        time.sleep(1.0)
                        self.route = self.ros_ctrl.getRoute()

                    if self.global_data.device_data.task_status == 0:
                        break

                    LogUtil.info(f"航线加载完成: {self.route}")

                    # 初始化本轮状态
                    self._reset_episode_state()
                    self.global_data.route = self.route
                    self.__reloadNavigationRoute(self.route)

                    # 本轮导航循环
                    self.episode_start_time = time.time()
                    self.episode_step_count = 0
                    self.timeout = False
                    last_update_metrics = None
                    for step in range(MAX_STEP_PER_EPISODE):
                        if self.global_data.device_data.task_status == 0:
                            LogUtil.info(f"步骤 {step} 停止训练")
                            break

                        if (time.time() - self.episode_start_time) > MAX_EPISODE_TIME:
                            LogUtil.info("本轮超时,提前结束")
                            self.timeout = True
                            break

                        self.episode_step_count = step + 1
                        self.done = self.navigationHandler(self.next_state, epoch, step)

                        # 定期更新PPO
                        if step > 0 and step % UPDATE_INTERVAL == 0:
                            last_update_metrics = self.ppo_agent.update()
                            self.training_logger.log_update(
                                epoch * MAX_STEP_PER_EPISODE + step,
                                last_update_metrics,
                            )

                        self.setMonitorParameterValue()

                        if self.done or self.arrive:
                            break

                        time.sleep(0.1)

                    if self.ppo_agent.buffer.rewards:
                        last_update_metrics = self.ppo_agent.update()
                        self.training_logger.log_update(
                            epoch * MAX_STEP_PER_EPISODE + self.episode_step_count,
                            last_update_metrics,
                        )

                    # 定期保存模型
                    if epoch % CHECKPOINT_INTERVAL == 0:
                        checkpoint_path = self.training_logger.checkpoint_dir / f"PPO_ship_obstacle_{epoch}.pth"
                        self.ppo_agent.save(checkpoint_path)
                        LogUtil.info(f"模型已保存: {checkpoint_path}")
                        self._save_dynamic_reward_weights()

                    self.training_logger.log_episode(
                        episode=epoch,
                        episode_return=self.episode_reward_sum,
                        steps=self.episode_step_count,
                        episode_time_sec=(time.time() - self.episode_start_time) if self.episode_start_time else 0.0,
                        arrived=self.arrive,
                        collided=self.done and not self.arrive,
                        timeout=self.timeout and not self.done and not self.arrive,
                        last_update_metrics=last_update_metrics,
                    )

            except Exception as e:
                LogUtil.error(e)
            finally:
                self.training_logger.close()
                time.sleep(2)

    def _reset_episode_state(self):
        """重置单轮训练的状态变量。"""
        self.episode_reward_sum = 0.0
        self.next_state = None
        self.last_distance = None
        self.done = False
        self.arrive = False
        self.timeout = False
        self.destPointIndex = -1
        self.destPoint = None
        self.max_distance = 0.0
        self.episode_step_count = 0
        self.apf_pid_integral = 0.0
        self.apf_pid_prev_error = 0.0

    def _build_fallback_route(self):
        """没有外部航线时，基于当前位置/航向生成一个临时航点，让船先动起来采集数据。"""
        pose = self.global_data.scada_data.pose
        lng = float(getattr(pose, "lng", 0.0) or 0.0)
        lat = float(getattr(pose, "lat", 0.0) or 0.0)
        yaw = float(getattr(pose, "yaw", 0.0) or 0.0)

        bearing_rad = math.radians(yaw)
        north_m = math.cos(bearing_rad) * FALLBACK_ROUTE_DISTANCE_M
        east_m = math.sin(bearing_rad) * FALLBACK_ROUTE_DISTANCE_M
        meters_per_deg_lat = 111320.0
        meters_per_deg_lng = max(1.0, meters_per_deg_lat * math.cos(math.radians(lat)))
        target_lat = lat + north_m / meters_per_deg_lat
        target_lng = lng + east_m / meters_per_deg_lng

        return DictToObject(**{
            "id": "fallback_collect_route",
            "name": "fallback_collect_route",
            "version": time.time(),
            "start_index": 0,
            "points": [{"lng": target_lng, "lat": target_lat, "high": 0.0, "speed": DEFAULT_SPEED}],
            "obstacles": [],
        })

    # ==================== 状态获取 ====================

    def getState(self, scan, heading: float, current_distance: float , degreeAship: float, current_speed: float, current_rotate_speed: float) -> list:
        """
        从激光雷达和传感器数据提取状态向量。

        Args:
            scan: 2D激光雷达数据对象(.ranges属性)
            heading: 船体航向(rad)
            current_distance: 到目标点的直线距离(m)
            degreeAship: 船对目标点的角度

        Returns:
            状态向量 [laser_features..., heading, distance, obstacle_min_range, obstacle_angle ,degreeAship]
        """

        heading_abs = self._normalize_heading_360(heading)
        target_abs = self._normalize_heading_360(degreeAship)
        angle_diff = self._normalize_signed_angle_diff(target_abs - heading_abs)

        scan_range = self._extract_laser_features(scan)
        obstacle_min_range, obstacle_angle = self._extract_obstacle_feature(
            scan_range=scan_range,
            angle_diff=angle_diff,
            current_distance=current_distance,
        )

        LogUtil.debug(
            f"状态: heading={heading:.2f}, distance={current_distance:.2f}, "
            f"obstacle_min={obstacle_min_range:.2f}, obstacle_angle={obstacle_angle}"
        )

        # 碰撞检测
        if COLLISION_DISTANCE > obstacle_min_range > 0:
            LogUtil.info(f"检测到碰撞! obstacle_min_range={obstacle_min_range}")
            self.done = True

        # 到达检测
        if self._is_last_waypoint_reached(current_distance):
            self.arrive = True

        return scan_range + [current_speed, current_rotate_speed, angle_diff, current_distance, obstacle_min_range, obstacle_angle]

    def _extract_laser_features(self, scan) -> list:
        """
        从激光雷达数据提取特征,仅取前方180°扇区用于碰撞检测。

        先将 90~180° 的数据接到 0~90° 前面，再从重排后的 0~180° 中按 2 步长采样，
        保持输出长度稳定为 90 个点。
        """
        scan_range = []
        ranges = list(scan.ranges)
        if len(ranges) >= 180:
            reordered_ranges = ranges[90:180] + ranges[0:90]
        else:
            reordered_ranges = ranges

        for i in range(0, min(len(reordered_ranges), 180), 2):
            value = reordered_ranges[i]
            if value == float('Inf') or value is None or np.isnan(value) or value > LASER_MAX_RANGE:
                scan_range.append(LASER_MAX_RANGE)
            else:
                scan_range.append(value)
        return scan_range if scan_range else [LASER_MAX_RANGE]

    def _extract_obstacle_feature(
        self,
        scan_range: list,
        angle_diff: float,
        current_distance: float,
    ) -> tuple[float, float]:
        """提取障碍物特征，并尽量排除目标点方向上的回波。"""
        if not scan_range:
            return LASER_MAX_RANGE, 0.0

        filtered_scan = list(scan_range)
        target_index = self._target_angle_to_scan_index(angle_diff, len(filtered_scan))
        if target_index is not None and current_distance > TARGET_ECHO_MIN_DISTANCE:
            window_bins = max(1, int(round(TARGET_ECHO_ANGLE_WINDOW_DEG / 2.0)))
            start = max(0, target_index - window_bins)
            end = min(len(filtered_scan), target_index + window_bins + 1)
            for idx in range(start, end):
                beam_range = filtered_scan[idx]
                if abs(beam_range - current_distance) <= TARGET_ECHO_RANGE_MARGIN:
                    filtered_scan[idx] = LASER_MAX_RANGE

        obstacle_idx = int(np.argmin(filtered_scan))
        obstacle_min_range = round(float(filtered_scan[obstacle_idx]), 2)

        # 如果整段都被屏蔽掉了，回退到原始最近点，避免状态失真。
        if obstacle_min_range >= LASER_MAX_RANGE:
            obstacle_idx = int(np.argmin(scan_range))
            obstacle_min_range = round(float(scan_range[obstacle_idx]), 2)

        return obstacle_min_range, float(obstacle_idx)

    def _target_angle_to_scan_index(self, angle_diff: float, scan_len: int) -> int | None:
        """将目标相对航向角映射到前向180度扫描索引。"""
        if scan_len <= 0:
            return None
        if angle_diff < -90.0 or angle_diff > 90.0:
            return None
        raw_index = int(round((angle_diff + 90.0) / 2.0))
        return max(0, min(scan_len - 1, raw_index))

    def _is_last_waypoint_reached(self, current_distance: float) -> bool:
        """判断是否到达最后一个航路点。"""
        return (
            self.destPointIndex + 1 == len(self.route.points)
            and current_distance < self.arrive_distance
        )

    # ==================== 动作执行 ====================

    def compute_bearing(self, lat1, lon1, lat2, lon2):
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lon = math.radians(lon2 - lon1)
        
        y = math.sin(delta_lon) * math.cos(lat2_rad)
        x = (math.cos(lat1_rad) * math.sin(lat2_rad) -
            math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon))
        bearing_rad = math.atan2(y, x)
        bearing_deg = (math.degrees(bearing_rad) + 360) % 360
        return bearing_deg

    @staticmethod
    def _normalize_heading_360(angle: float) -> float:
        return angle % 360

    @staticmethod
    def _normalize_signed_angle_diff(angle: float) -> float:
        return (angle + 180) % 360 - 180

    def _calc_distance_to_target(self, lng1: float, lat1: float, lng2: float, lat2: float) -> float:
        rad_lat1 = math.radians(lat1)
        rad_lat2 = math.radians(lat2)
        delta_lat = rad_lat1 - rad_lat2
        delta_lng = math.radians(lng1) - math.radians(lng2)
        distance = 2 * math.asin(
            math.sqrt(
                math.sin(delta_lat / 2) ** 2
                + math.cos(rad_lat1) * math.cos(rad_lat2) * math.sin(delta_lng / 2) ** 2
            )
        )
        return round(distance * 6378.137 * 1000, 1)

    def step(self, state: list, action: np.ndarray, laser_scan, heading: float,
             shipToNextWPDistance: float,degreeAship: float, max_distance: float,prev_distance: float) -> StepResult:
        """
        执行动作并获取环境反馈。

        Returns:
            StepResult: 包含下一状态、奖励、控制指令等
        """
        pose = self.global_data.scada_data.pose

        if self.destPoint is not None:
            degree_text = f" degreeAship={degreeAship:.9f} "
            print(
                f"target=({self.destPoint.lng:.6f},{self.destPoint.lat:.6f}) "
                f"ship=({pose.lng:.6f},{pose.lat:.6f}){degree_text}"
            )
        obstacle_min_range = state[-2]
        current_distance = state[-3]

        # 动作到控制量的映射：APF-PID 提供稳定先验，PPO 学习残差修正
        adviseRotate, adviseSpeed = self._action_to_control(
            action,
            obstacle_min_range,
            current_distance,
            angle_diff=state[-4],
            obstacle_angle=state[-1],
            heading_world=heading,
            target_heading_world=degreeAship,
        )

        # 获取新状态
        pose = self.global_data.scada_data.pose
        new_state = self.getState(laser_scan, heading, shipToNextWPDistance, degreeAship, pose.speed, pose.rotate_speed)

        # 使用分解函数
        breakdown = compute_reward_breakdown(
            state=new_state,
            action=action,
            max_distance=max_distance,
            angle_diff=new_state[-4],
            arrive=self.arrive,
            done=self.done,
            prev_state=state,
            prev_distance=prev_distance,
            heading_world=heading,
            target_heading_world=degreeAship,
            episode_elapsed_time=time.time() - self.episode_start_time,
            config=self.reward_config,
        )
        task_reward = self._compute_discrete_task_reward()
        reward, dynamic_weights, update_metrics = self._compute_dynamic_reward(
            breakdown=breakdown,
            distance=shipToNextWPDistance,
            degree_aship=degreeAship,
            obstacle_min_range=state[-2],
            obstacle_angle=state[-1],
            task_reward=task_reward,
            done_flag=self.done or self.arrive,
        )
        self._update_apf_debug_view(
            prev_state=state,
            new_state=new_state,
            heading=heading,
            target_heading_world=degreeAship,
            action=action,
            reward=reward,
        )

        # reward = compute_reward(
        #     state=new_state,
        #     action=action,
        #     max_distance=max_distance,
        #     angle_diff=new_state[-4],
        #     arrive=self.arrive,
        #     done=self.done,
        #     config=self.reward_config,
        # )
        if self.arrive:
            LogUtil.info("到达目标!")
        elif self.done:
            LogUtil.info("发生碰撞!")

        return StepResult(
            next_state=np.asarray(new_state),
            reward=reward,
            current_distance=current_distance,
            advise_speed=adviseSpeed,
            advise_rotate=adviseRotate,
            advised_heading=heading,
            distance=shipToNextWPDistance,
            degree_aship=degreeAship,
            obstacle_min_range=state[-2],
            obstacle_angle=state[-1],
            breakdown=breakdown,
            task_reward=task_reward,
            dynamic_weights=dynamic_weights,
            reward_weight_update_metrics=update_metrics,
        )

    def _compute_discrete_task_reward(self) -> float:
        if self.arrive:
            return 1.0
        if self.done:
            return -1.0
        if self.timeout:
            return -0.3
        return -0.01

    def _compute_dynamic_reward(
        self,
        breakdown: RewardBreakdown,
        distance: float,
        degree_aship: float,
        obstacle_min_range: float,
        obstacle_angle: float,
        task_reward: float,
        done_flag: bool,
    ) -> tuple[float, tuple[float, float, float, float] | None, dict[str, float] | None]:
        if self.reward_weight_updater is None or self.reward_normalizer is None:
            return breakdown.total_reward, None, None

        raw_step = {
            "distance": distance,
            "degreeAship": degree_aship,
            "obstacle_min_range": obstacle_min_range,
            "obstacle_angle": obstacle_angle,
            "distance_reward_raw": breakdown.distance_raw,
            "heading_reward_raw": breakdown.heading_raw,
            "obstacle_reward_raw": breakdown.obstacle_raw,
            "task_reward": task_reward,
            "done": float(done_flag),
        }
        try:
            features, reward_components, online_task_reward, online_done = raw_step_to_network_input(
                raw_step,
                self.reward_normalizer,
            )
            weights = self.reward_weight_updater.get_weights(features)
            continuous_reward = float(
                weights[0] * reward_components[0].item()
                + weights[1] * reward_components[1].item()
                + weights[2] * reward_components[2].item()
                + weights[3] * reward_components[3].item()
            )
            scaled_task_reward = online_task_reward * REWARD_TASK_SCALE
            weighted_reward = float(np.clip(
                continuous_reward * REWARD_CONTINUOUS_SCALE + scaled_task_reward,
                FINAL_REWARD_MIN,
                FINAL_REWARD_MAX,
            ))
            self.reward_weight_updater.add_experience(
                features,
                reward_components,
                scaled_task_reward,
                online_done,
            )
            update_metrics = self.reward_weight_updater.maybe_update()
            if update_metrics is not None:
                LogUtil.info(
                    "奖励权重在线更新: "
                    f"loss={update_metrics['loss']:.4f}, corr={update_metrics['corr']:.4f}, "
                    f"buffer={update_metrics['buffer_size']:.0f}, "
                    f"update_times={update_metrics['update_times']:.0f}"
                )
                if REWARD_WEIGHT_SAVE_ON_UPDATE:
                    self._save_dynamic_reward_weights()
            return weighted_reward, weights, update_metrics
        except Exception as exc:
            LogUtil.error(f"动态奖励计算失败，回退固定奖励: {exc}")
            return breakdown.total_reward, None, None

    def _save_dynamic_reward_weights(self):
        if self.reward_weight_updater is None:
            return
        try:
            save_reward_net(self.reward_weight_updater.reward_net, str(REWARD_WEIGHT_MODEL_PATH))
            LogUtil.info(f"动态奖励权重已保存: {REWARD_WEIGHT_MODEL_PATH}")
        except Exception as exc:
            LogUtil.error(f"动态奖励权重保存失败: {exc}")

    def _log_reward_weights(
        self,
        episode: int,
        step: int,
        result: StepResult,
        update_metrics: dict[str, float] | None,
    ) -> None:
        """把动态权重输出到CSV，方便直接查看每步的权重变化。"""
        if result.dynamic_weights is None:
            return

        weight_0, weight_1, weight_2, weight_3 = result.dynamic_weights
        metrics = update_metrics or result.reward_weight_update_metrics or {}

        with open(self.reward_weight_log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                episode,
                step,
                self.episode_step_count,
                f"{result.distance:.3f}",
                f"{result.current_distance:.3f}",
                f"{weight_0:.6f}",
                f"{weight_1:.6f}",
                f"{weight_2:.6f}",
                f"{weight_3:.6f}",
                f"{result.reward:.6f}",
                f"{metrics.get('loss', '')}",
                f"{metrics.get('corr', '')}",
                f"{metrics.get('buffer_size', '')}",
                f"{metrics.get('update_times', '')}",
            ])

    def _action_to_control(
        self,
        action: np.ndarray,
        obstacle_min_range: float,
        current_distance: float,
        angle_diff: float | None = None,
        obstacle_angle: float | None = None,
        heading_world: float | None = None,
        target_heading_world: float | None = None,
    ) -> tuple:
        if HAS_CONTINUOUS_ACTION:
            """连续动作：APF-PID 基础控制 + PPO 残差控制。"""
            action_vec = np.asarray(action, dtype=np.float32).reshape(-1)
            if action_vec.size < 2:
                raise ValueError(f"continuous action must have at least 2 elements, got shape={np.asarray(action).shape}")
            turn_ratio = float(np.clip(action_vec[0], -1.0, 1.0))
            speed_ratio = float(np.clip(action_vec[1], -1.0, 1.0))

            apf_pid_ratio = self._compute_apf_pid_turn_ratio(
                angle_diff=angle_diff if angle_diff is not None else 0.0,
                current_distance=current_distance,
                obstacle_min_range=obstacle_min_range,
                obstacle_angle=obstacle_angle if obstacle_angle is not None else 0.0,
                heading_world=heading_world,
                target_heading_world=target_heading_world,
            )
            ppo_blend = self._compute_ppo_blend(obstacle_min_range, current_distance)
            final_turn_ratio = float(np.clip((1.0 - ppo_blend) * apf_pid_ratio + ppo_blend * turn_ratio, -1.0, 1.0))
            adviseRotate = round(final_turn_ratio * ANGULAR_VELOCITY_MAX, 0)

            # 自适应速度
            base_speed = (speed_ratio + 1.0) * 0.5 * ACTION_TO_SPEED_CONTINOUS_SCALE
            adviseSpeed = max(0.0, min(ACTION_TO_SPEED_CONTINOUS_SCALE, base_speed))

            if obstacle_min_range < 1.0:
                adviseSpeed = min(adviseSpeed, ACTION_TO_SPEED_CONTINOUS_SCALE * 0.2)
            elif obstacle_min_range < 2.0:
                adviseSpeed = min(adviseSpeed, ACTION_TO_SPEED_CONTINOUS_SCALE * 0.4)

            if current_distance < TARGET_SLOW_RANGE:
                adviseSpeed = min(adviseSpeed, ACTION_TO_SPEED_CONTINOUS_SCALE * 0.3)

        else:
            """将离散动作映射为(转向百分比, 速度百分比)。"""
            # 角度计算：将action映射到[-100, +100]度范围
            ang_vel = ((self.action_size - 1) / 2 - action) * ANGULAR_VELOCITY_MAX / ((self.action_size - 1) / 2)
            adviseRotate = round(ang_vel, 0) * ACTION_TO_DEGREE_CONTINOUS_SCALE
             # 自适应速度
            adviseSpeed = DEFAULT_SPEED
            if obstacle_min_range < OBSTACLE_SLOW_RANGE:
                adviseSpeed = 0.1
            if current_distance < TARGET_SLOW_RANGE:
                adviseSpeed = 0.1
            adviseSpeed = min(round(adviseSpeed * 100 / DEFAULT_SPEED, 0), 100)

        return adviseRotate, adviseSpeed

    def _compute_apf_pid_turn_ratio(
        self,
        angle_diff: float,
        current_distance: float,
        obstacle_min_range: float,
        obstacle_angle: float,
        heading_world: float | None,
        target_heading_world: float | None,
    ) -> float:
        apf_error_deg = calc_apf_heading_diff(
            angle_diff=angle_diff,
            current_distance=current_distance,
            obstacle_min_range=obstacle_min_range,
            obstacle_angle=obstacle_angle,
            heading_world=heading_world,
            target_heading_world=target_heading_world,
            config=self.reward_config,
        )
        error = max(-1.0, min(1.0, apf_error_deg / 90.0))
        self.apf_pid_integral = max(
            -APF_PID_MAX_INTEGRAL,
            min(APF_PID_MAX_INTEGRAL, self.apf_pid_integral + error * CONTROL_DT),
        )
        derivative = (error - self.apf_pid_prev_error) / max(CONTROL_DT, 1e-6)
        self.apf_pid_prev_error = error
        return float(np.clip(
            APF_PID_KP * error + APF_PID_KI * self.apf_pid_integral + APF_PID_KD * derivative,
            -1.0,
            1.0,
        ))

    @staticmethod
    def _compute_ppo_blend(obstacle_min_range: float, current_distance: float) -> float:
        obstacle_risk = max(0.0, min(1.0, (OBSTACLE_SLOW_RANGE - obstacle_min_range) / OBSTACLE_SLOW_RANGE))
        target_risk = max(0.0, min(1.0, (TARGET_SLOW_RANGE - current_distance) / TARGET_SLOW_RANGE))
        expert_ratio = max(obstacle_risk, target_risk)
        return max(PPO_MIN_BLEND, min(1.0, 1.0 - 0.55 * expert_ratio))

    @staticmethod
    def _sim_deg_to_canvas_deg(angle_deg: float) -> float:
        """将仿真角度转换为画布角度：仿真0°朝上，画布0°朝右。"""
        return (90.0 - angle_deg) % 360.0

    @staticmethod
    def _world_deg_to_canvas_point(center: tuple[int, int], length: float, angle_deg: float) -> tuple[int, int]:
        angle_rad = math.radians(angle_deg)
        x = center[0] + length * math.cos(angle_rad)
        y = center[1] - length * math.sin(angle_rad)
        return int(round(x)), int(round(y))

    def _draw_debug_arrow(
        self,
        canvas: np.ndarray,
        center: tuple[int, int],
        length: float,
        angle_deg: float,
        color: tuple[int, int, int],
        label: str,
    ) -> None:
        canvas_angle = self._sim_deg_to_canvas_deg(angle_deg)
        end = self._world_deg_to_canvas_point(center, length, canvas_angle)
        cv2.arrowedLine(canvas, center, end, color, 2, tipLength=0.14)
        text_pos = self._world_deg_to_canvas_point(center, length + 18, canvas_angle)
        cv2.putText(canvas, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    def _draw_debug_point(
        self,
        canvas: np.ndarray,
        center: tuple[int, int],
        distance_m: float,
        angle_deg: float,
        color: tuple[int, int, int],
        radius: int = 3,
    ) -> None:
        """按仿真角度和距离，在画布上画出一个采样点。"""
        canvas_angle = self._sim_deg_to_canvas_deg(angle_deg)
        point_radius = min(220.0, 30.0 + float(distance_m) * 45.0)
        point = self._world_deg_to_canvas_point(center, point_radius, canvas_angle)
        cv2.circle(canvas, point, radius, color, -1)

    def _update_apf_debug_view(
        self,
        prev_state: list,
        new_state: list,
        heading: float,
        target_heading_world: float,
        action: np.ndarray,
        reward: float,
    ) -> None:
        if not self.enable_apf_debug_view:
            return

        try:
            heading_world = self._normalize_heading_360(heading)
            target_world = self._normalize_heading_360(target_heading_world)
            prev_apf_heading_diff = calc_apf_heading_diff(
                angle_diff=prev_state[-4],
                current_distance=prev_state[-3],
                obstacle_min_range=prev_state[-2],
                obstacle_angle=prev_state[-1],
                heading_world=heading_world,
                target_heading_world=target_world,
                config=self.reward_config,
            )
            curr_apf_heading_diff = calc_apf_heading_diff(
                angle_diff=new_state[-4],
                current_distance=new_state[-3],
                obstacle_min_range=new_state[-2],
                obstacle_angle=new_state[-1],
                heading_world=heading_world,
                target_heading_world=target_world,
                config=self.reward_config,
            )

            action_vec = np.asarray(action, dtype=np.float32).reshape(-1)
            turn_action = float(action_vec[0]) if action_vec.size > 0 else 0.0
            predicted_apf_heading_diff = self._normalize_signed_angle_diff(
                curr_apf_heading_diff - turn_action * ANGULAR_VELOCITY_MAX * CONTROL_DT
            )

            canvas = np.full((720, 760, 3), 248, dtype=np.uint8)
            center = (380, 380)
            cv2.circle(canvas, center, 230, (225, 225, 225), 1)
            cv2.circle(canvas, center, 4, (30, 30, 30), -1)

            self._draw_debug_arrow(canvas, center, 130, heading_world, (50, 50, 50), "ship")
            self._draw_debug_arrow(canvas, center, 180, target_world, (40, 160, 40), "target")
            self._draw_debug_arrow(canvas, center, 220, heading_world + curr_apf_heading_diff, (30, 80, 220), "apf_now")
            self._draw_debug_arrow(canvas, center, 150, heading_world + predicted_apf_heading_diff, (180, 60, 180), "apf_pred")

            laser_count = max(0, len(new_state) - 6)
            laser_points = np.asarray(new_state[:laser_count], dtype=np.float32)
            if laser_points.size > 0:
                obstacle_idx = int(np.argmin(laser_points))
                for idx, beam_range in enumerate(laser_points):
                    # 源代码中的前方180°扇区按2°采样，这里按索引还原角度。
                    relative_deg = -90.0 + idx * 2.0
                    point_color = (30, 30, 200) if idx == obstacle_idx else (120, 120, 120)
                    point_radius = 4 if idx == obstacle_idx else 2
                    self._draw_debug_point(
                        canvas,
                        center,
                        float(beam_range),
                        heading_world + relative_deg,
                        point_color,
                        radius=point_radius,
                    )

                obstacle_angle_idx = float(new_state[-1])
                obstacle_relative_deg = obstacle_angle_idx * 2.0 - 90.0
                if 0 <= int(round(obstacle_angle_idx)) < laser_points.size:
                    obstacle_point_range = float(laser_points[int(round(obstacle_angle_idx))])
                    self._draw_debug_point(
                        canvas,
                        center,
                        obstacle_point_range,
                        heading_world + obstacle_relative_deg,
                        (20, 140, 255),
                        radius=5,
                    )
                    obstacle_point = self._world_deg_to_canvas_point(
                        center,
                        min(220.0, 30.0 + obstacle_point_range * 45.0),
                        self._sim_deg_to_canvas_deg(heading_world + obstacle_relative_deg),
                    )
                    cv2.putText(
                        canvas,
                        "obstacle_angle",
                        (obstacle_point[0] + 10, obstacle_point[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (20, 140, 255),
                        1,
                        cv2.LINE_AA,
                    )

            debug_lines = [
                f"heading_world: {heading_world:.2f} deg",
                f"target_heading_world: {target_world:.2f} deg",
                f"prev_angle_diff: {prev_state[-4]:.2f} deg",
                f"curr_angle_diff: {new_state[-4]:.2f} deg",
                f"obstacle_angle(index): {new_state[-1]:.2f}",
                f"obstacle_angle(relative): {obstacle_relative_deg:.2f} deg",
                f"obstacle_min_range: {new_state[-2]:.2f} m",
                f"prev_apf_heading_diff: {prev_apf_heading_diff:.2f} deg",
                f"curr_apf_heading_diff: {curr_apf_heading_diff:.2f} deg",
                f"pred_apf_heading_diff: {predicted_apf_heading_diff:.2f} deg",
                f"laser_points: {laser_count}",
                f"turn_action: {turn_action:.3f}",
                f"reward: {reward:.3f}",
            ]
            for idx, line in enumerate(debug_lines):
                cv2.putText(
                    canvas,
                    line,
                    (24, 32 + idx * 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (20, 20, 20),
                    2,
                    cv2.LINE_AA,
                )

            cv2.putText(
                canvas,
                "sim 0 deg -> up, canvas 0 deg -> right",
                (24, 695),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (90, 90, 90),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow(APF_DEBUG_WINDOW_NAME, canvas)
            cv2.waitKey(1)
        except Exception as exc:
            self.enable_apf_debug_view = False
            LogUtil.error(f"APF debug view disabled: {exc}")

    # ==================== 奖励函数 ====================

    def _compute_reward(self, state: list, action: np.ndarray, max_distance: float, angle_diff: float) -> float:
        """计算综合奖励值。"""
        obstacle_min_range = state[-2]
        current_distance = state[-3]
        angle_diff_state = state[-4]

        # 距离奖励
        distance_reward = self._calc_distance_reward(current_distance, max_distance)

        # 更新最大距离记录
        if current_distance > max_distance:
            self.max_distance = current_distance

        # 航向奖励
        heading_reward = self._calc_heading_reward(action, angle_diff_state, current_distance, max_distance, angle_diff)

        # 障碍物/接近目标奖励
        obstacle_reward = 0.0
        if obstacle_min_range < 3:
            obstacle_reward = REWARD_OBSTACLE_PENALTY
        elif current_distance < TARGET_SLOW_RANGE:
            obstacle_reward = REWARD_NEAR_TARGET_BONUS

        # 加权组合
        reward = (
            distance_reward * REWARD_WEIGHT_DISTANCE
            + obstacle_reward * REWARD_WEIGHT_OBSTACLE
            + heading_reward * REWARD_WEIGHT_HEADING
        )

        # 终止奖励/惩罚
        if self.arrive:
            LogUtil.info("到达目标!")

            # 计算自此轮训练开始的时间
            episode_elapsed_time = time.time() - self.episode_start_time
            # 时间奖励权重
            time_reward_weight = self._calc_time_reward_weight(episode_elapsed_time)

            reward += REWARD_ARRIVE_BONUS * time_reward_weight
        elif self.done:
            LogUtil.info("发生碰撞!")
            reward += REWARD_COLLISION_PENALTY

        return reward

    @staticmethod
    def _calc_time_reward_weight(episode_elapsed_time: float) -> float:
        """计算时间权重。时间等于 MAX_EPISODE_TIME 时最小，为 0.5；等于 0 时最大，为 1.0。"""
        if MAX_EPISODE_TIME <= 0:
            return 1.0

        progress = episode_elapsed_time / MAX_EPISODE_TIME
        progress = max(0.0, min(1.0, progress))

        min_weight = 0.5
        return min_weight + (1.0 - min_weight) * ((1.0 - progress) ** 2)

    @staticmethod
    def _calc_distance_reward(current_distance: float, max_distance: float) -> float:
        """计算基于目标距离的奖励。"""
        if current_distance <= 1:
            return 0.0
        reward = 1 - (current_distance / max_distance)
        return reward * 2 if reward < 0 else reward * 5

    @staticmethod
    def _calc_heading_reward(action: np.ndarray, angle_diff_state: float,
                             current_distance: float, max_distance: float, angle_diff: float) -> float:
        distance_rate = 2 ** (current_distance / max_distance) if max_distance > 0 else 1.0
        if not HAS_CONTINUOUS_ACTION:
            """计算航向对齐奖励。"""
            yaw_rewards = []
            pi = math.pi
            for i in range(N_ACTIONS):
                angle = -pi / 4 + angle_diff_state + (pi / 8 * i) + pi / 2
                tr = 1 - 4 * abs(0.5 - math.modf(0.25 + 0.5 * angle % (2 * pi) / pi)[0])
                yaw_rewards.append(tr)

            return round(yaw_rewards[action[0]] * 5, 2) * distance_rate
        else:
            """连续动作的航向奖励"""
            angular_speed = action[0] * ANGULAR_VELOCITY_MAX          # 度/秒
            predicted_angle_diff = (angle_diff - angular_speed * CONTROL_DT + 180) % 360 - 180
            heading_reward = 1 - 2 * (abs(predicted_angle_diff) / 180.0)  # [1,-1]

            return round(heading_reward, 2)

    # ==================== 导航处理 ====================

    def navigationHandler(self, state, episode: int, step: int) -> bool:
        """
        导航处理主入口。
        Returns:
            bool: 是否终止本轮导航
        """
        try:
            # 1. 获取载体姿态
            pose_info = self._load_vehicle_pose_info()

            # 2. 路径规划
            nav_context = self._update_navigation_target(pose_info)
            if nav_context is None:
                return False

            # 3. 获取激光雷达数据
            laser_scan = self._wait_for_laser_data()
            if laser_scan is None:
                return False

            # 4. 执行PPO决策循环
            done = self._ppo_decision_loop(state, episode, step, laser_scan, nav_context)

            # 5. 日志输出
            self._log_navigation_result(nav_context)

            return done

        except Exception as e:
            LogUtil.error(f"navigationHandler异常: {e}")
            return False

    def _load_vehicle_pose_info(self) -> tuple:
        """获取载体位置和姿态信息。"""
        workModel = self.global_data.device_data.work_model
        isReturn = workModel == Constants.WorkMode.AutoReturn
        isAuto = workModel == Constants.WorkMode.Auto

        pose = self.global_data.scada_data.pose
        lng, lat = pose.lng, pose.lat
        heading = pose.yaw
        if heading > 180:
            heading -= 360

        return isAuto, isReturn, lng, lat, heading, pose.speed, pose.rotate_speed

    def _update_navigation_target(self, pose_info: tuple):
        """更新路径规划目标点,返回导航上下文。"""
        isAuto, isReturn, lng, lat, _, _, _ = pose_info
        self.routePlaneService.setCurrentPos(lng, lat, isReturn)

        try:
            nextPointIndex = self.routePlaneService.curNextIndex
            prevPointIndex = self.routePlaneService.curPrevIndex
            nextPoint = self.route.points[nextPointIndex]

            if prevPointIndex < 0 or prevPointIndex >= len(self.route.points):
                prevPointIndex = nextPointIndex

            if nextPointIndex != self.destPointIndex or nextPoint != self.destPoint:
                self.max_distance = 0.0
                LogUtil.info(f"切换至航点[{nextPointIndex}]={nextPoint}, 距离={self.max_distance:.1f}m")

            self.destPointIndex = nextPointIndex
            self.destPoint = self.route.points[self.destPointIndex]
            self.prevPointIndex = prevPointIndex
            self.prevPoint = self.route.points[prevPointIndex]

            #现在的位置到目标点的距离和角度都是手算
            shipToNextWPDistance = self._calc_distance_to_target(lng, lat, self.destPoint.lng, self.destPoint.lat)
            degreeAship = self.compute_bearing(lat, lng, self.destPoint.lat, self.destPoint.lng)
            if self.max_distance <= 0:
                self.max_distance = shipToNextWPDistance

        except Exception as e:
            LogUtil.error(f"路径规划异常: {e}")

        if self.destPointIndex == -1:
            return None

        return {
            'nextPointIndex': nextPointIndex,
            'shipToNextWPDistance': shipToNextWPDistance,
            'degreeAship' : degreeAship
        }

    def _wait_for_laser_data(self):
        """等待新的激光雷达数据,超时返回None。"""
        laser_scan = self.get_laser_scan(timeout=2)
        if laser_scan is None:
            LogUtil.info("获取激光雷达数据超时(2s)")
            return None
        self.last_laser_scan = laser_scan
        return laser_scan

    def _ppo_decision_loop(self, state, episode: int, step: int,
                       laser_scan, nav_context: dict) -> bool:
        """PPO核心决策逻辑。"""
        heading = self._get_current_heading()
        pose = self.global_data.scada_data.pose
        current_speed = pose.speed
        current_rotate_speed = pose.rotate_speed

        # 初始化状态
        if state is None:
            state = self.getState(laser_scan, heading, nav_context['shipToNextWPDistance'],
                                nav_context['degreeAship'], current_speed, current_rotate_speed)

        # 获取上一步距离（用于进度奖励）
        current_distance = state[-3] if state is not None else nav_context['shipToNextWPDistance']
        prev_distance = self.last_distance if self.last_distance is not None else current_distance

        state_tensor = torch.FloatTensor(state).to(device)
        action = self.ppo_agent.select_action(state_tensor)

        # 执行动作
        result = self.step(
            state_tensor.cpu().numpy().tolist(), action,
            laser_scan, heading, nav_context['shipToNextWPDistance'],
            nav_context['degreeAship'], self.max_distance, prev_distance
        )
        self.next_state = result.next_state
        self.last_distance = result.distance   # 保存当前步距离供下一步使用

        # 记录经验
        self.ppo_agent.buffer.rewards.append(result.reward)
        self.ppo_agent.buffer.is_terminals.append(self.done)
        self.episode_reward_sum += result.reward
        self._log_reward_weights(episode, step, result, result.reward_weight_update_metrics)

        # ---------- 数据收集 CSV ----------
        if self.data_collection_enabled and self.data_collection_count < self.data_collection_max:
            done_flag = self.done or self.arrive
            discrete_reward = self._compute_discrete_csv_reward(result)
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    result.distance,
                    result.degree_aship,
                    result.obstacle_min_range,
                    result.obstacle_angle,
                    result.breakdown.distance_raw,
                    result.breakdown.heading_raw,
                    result.breakdown.obstacle_raw,
                    discrete_reward,
                    int(done_flag)
                ])
            self.data_collection_count += 1
            if self.data_collection_count >= self.data_collection_max:
                self._finish_data_collection()
        # ---------------------------------

        # 输出控制量（修复点：补全参数）
        self._output_control_commands(result, episode, step, nav_context['nextPointIndex'])

        LogUtil.debug(
            f"Step={step} Action={action} Reward={result.reward:.2f} "
            f"Done={self.done} Arrive={self.arrive}"
        )

        if self.done:
            return True
        if self.arrive:
            LogUtil.info(f"Episode结束于step={step}, 总奖励={self.episode_reward_sum:.1f}")
            return True
        return False

    def _get_current_heading(self) -> float:
        """获取当前航向。"""
        pose = self.global_data.scada_data.pose
        heading = pose.yaw
        return heading - 360 if heading > 180 else heading

    def _output_control_commands(self, result: StepResult, episode: int, step: int, nextPointIndex: int):
        """将控制命令输出到global_data。"""
        self.global_data.updateAlgorithmOutput(
            episode, step, int(self.episode_reward_sum),
            result.reward, MAX_EPOCH, 2
        )
        self.global_data.updateThrottleRudderOutput(
            result.advise_speed, result.advise_rotate, result.advised_heading,
            nextPointIndex, result.current_distance
        )

    def _compute_discrete_csv_reward(self, result: StepResult) -> float:
        """计算CSV/离线训练用的任务级离散奖励。"""
        if self.arrive:
            return REWARD_ARRIVE_BONUS
        if self.done:
            return REWARD_COLLISION_PENALTY
        if result.obstacle_min_range < 3:
            return REWARD_OBSTACLE_PENALTY
        if result.distance < TARGET_SLOW_RANGE:
            return REWARD_NEAR_TARGET_BONUS
        return 0.0

    def _finish_data_collection(self):
        """CSV采集完成后停止采集，并自动打开结果文件。"""
        if self.data_collection_completed:
            return

        self.data_collection_enabled = False
        self.data_collection_completed = True
        csv_path = self.csv_path.resolve()
        LogUtil.info(f"数据收集完成，已保存 {self.data_collection_count} 条记录至 {csv_path}")

        if AUTO_TRAIN_REWARD_NET_AFTER_COLLECTION and self.reward_weight_updater is None:
            self._start_reward_weight_training_thread()

        try:
            if hasattr(os, "startfile"):
                os.startfile(str(csv_path))
            else:
                LogUtil.info(f"请手动打开CSV文件: {csv_path}")
        except Exception as e:
            LogUtil.error(f"自动打开CSV失败，请手动打开 {csv_path}: {e}")

    def _start_reward_weight_training_thread(self):
        """采集到足够CSV后，后台训练奖励权重网络，训练完成后立即启用动态权重。"""
        if self.reward_weight_training:
            return
        self.reward_weight_training = True

        def _worker():
            try:
                LogUtil.info("开始用 data_collection.csv 训练动态奖励权重网络...")
                train_reward_from_c4.main()
                self._init_dynamic_reward_weights()
                if self.reward_weight_updater is not None:
                    LogUtil.info("动态奖励权重网络训练完成，已在当前运行中启用")
                else:
                    LogUtil.error("动态奖励权重网络训练结束，但加载失败，请检查 pth/json 输出")
            except Exception as exc:
                LogUtil.error(f"动态奖励权重网络自动训练失败: {exc}")
            finally:
                self.reward_weight_training = False

        threading.Thread(target=_worker, daemon=True).start()

    def _log_navigation_result(self, nav_context: dict):
        """记录导航结果日志。"""
        LogUtil.info(
            f"航点={self.destPointIndex} 距离={nav_context['shipToNextWPDistance']:.1f}m "
            f"速度=... 转向=..."
        )

    # ==================== 参数注册 ====================

    def registerParameter(self):
        """注册可调参数到ROS2参数服务器。"""
        self.ros_ctrl.initParameterList()

        params = [
            Parameter(name='/usv/auto/plan/attainRadius', dataType='float', defaultData=2),
            Parameter(name='/usv/auto/param2', dataType='int', defaultData=[20]),
            Parameter(name='/usv/auto/param3', dataType='int', defaultData=[30, 31]),
            Parameter(name='/usv/auto/param4', dataType='float', defaultData=[40.1, 40.2]),
            Parameter(name='/usv/auto/param6', dataType='str', defaultData='some msg'),
        ]

        for param in params:
            if self.ros_ctrl.registParameter(param):
                self.global_data.parameterAdjustMap.update({param.name: param})
                if param.name == '/usv/auto/plan/attainRadius':
                    self.routePlaneService.wayPointRadius = param.value

    def setMonitorParameterValue(self):
        """监控参数回调(待实现)。"""
        pass

    # ==================== 内部工具方法 ====================

    def __loadVehiclePoseInfo(self):
        """向后兼容接口。"""
        return self._load_vehicle_pose_info()

    def __reloadNavigationRoute(self, route) -> bool:
        """重新加载导航航线。"""
        self.route = route
        self.routePlaneService.reset(route=self.route)
        return True

    def get_laser_scan(self, timeout: float):
        """等待新的激光雷达数据,超时返回None。"""
        laser_start_time = time.time()
        while self.global_data.laser_data == self.last_laser_scan:
            time.sleep(0.1)
            if (time.time() - laser_start_time) > timeout:
                return None
        return self.global_data.laser_data
    
    # =================== CSV工具 ====================

    def _init_csv_file(self):
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "distance", "degreeAship", "obstacle_min_range", "obstacle_angle",
                "distance_reward_raw", "heading_reward_raw", "obstacle_reward_raw",
                "task_reward", "done"
            ])

    def _init_reward_weight_log_file(self):
        try:
            f = open(self.reward_weight_log_path, 'w', newline='', encoding='utf-8')
        except PermissionError:
            self.reward_weight_log_path = self.reward_weight_log_path.with_name(
                f"{self.reward_weight_log_path.stem}_{int(time.time())}{self.reward_weight_log_path.suffix}"
            )
            LogUtil.error(f"reward_weight_output.csv 被占用，改写到 {self.reward_weight_log_path}")
            f = open(self.reward_weight_log_path, 'w', newline='', encoding='utf-8')

        with f:
            writer = csv.writer(f)
            writer.writerow([
                "episode",
                "step",
                "episode_step_count",
                "distance",
                "current_distance",
                "w_goal_progress",
                "w_goal_proximity",
                "w_obs_dist",
                "w_apf",
                "reward",
                "loss",
                "corr",
                "buffer_size",
                "update_times",
            ])

    def close(self):
        if getattr(self, "enable_apf_debug_view", False):
            cv2.destroyAllWindows()
        self._save_dynamic_reward_weights()
        if hasattr(self, "training_logger") and self.training_logger is not None:
            self.training_logger.close()


# 向后兼容：保留旧类名供main.py等外部引用
DQN_NAV = PPONav
