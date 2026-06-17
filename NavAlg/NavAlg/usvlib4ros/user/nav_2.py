import math
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

import numpy as np
import torch

from usvlib4ros.navigation.usv_ros2_controller import Ros2Controller
from usvlib4ros.navigation.route_plan_service import RoutePlanService
from usvlib4ros.msg.global_data import GlobalData, DictToObject, Point, Constants
from usvlib4ros.msg.parameter import Parameter
from usvlib4ros.usvRosUtil import LogUtil
from usvlib4ros.user.episode_logging import EpisodeDataLogger
from usvlib4ros.user.pid_controller import PID, PIDConfig
from usvlib4ros.user.PP0_2 import PPO, device
from usvlib4ros.user.reward import RewardConfig, compute_reward_breakdown, RewardBreakdown, calc_apf_heading_diff
from usvlib4ros.user.tensorboard_logging import TensorBoardMetricsWriter, build_tensorboard_log_dir
from usvlib4ros.user.training_logger import TrainingLogger

# ==================== 超参数配置 ====================
N_ACTIONS = 2          # 连续动作空间
HAS_CONTINUOUS_ACTION = True  # 是否使用连续动作空间
N_STATES = 96          # 状态维度: 前方180°激光点数(约90) + speed + rotated_speed + angel_diff + distance + obstacle_min_range + obstacle_angle
MEMORY_CAPACITY = 2000
BATCH_SIZE = 128
LR_ACTOR = 0.0003
LR_CRITIC = 0.0001
GAMMA = 0.99           # 折扣因子
K_EPOCHS = 20          # PPO更新轮数
EPS_CLIP = 0.2         # PPO裁剪系数
ACTION_STD_INIT = 1  # 连续动作标准差初始化(当前未启用连续空间)
MAX_EPOCH = 4000       # 最大训练轮数
MAX_STEP_PER_EPISODE = 500   # 每轮最大步数
MAX_EPISODE_TIME = 300  # 每轮最大时间(秒)
UPDATE_INTERVAL = 200  # PPO更新间隔(步数)
MIN_BUFFER_SIZE_FOR_UPDATE = 128
CHECKPOINT_INTERVAL = 100  # 模型保存间隔(轮数)
RESET_SETTLE_DELAY = 3.0  # 每轮复位后等待仿真刷新(秒)
MAX_RESET_RETRIES = 5
MIXED_ROTATE_CONTROL = False  # 是否启用APF-PID与PPO混合切换
TEACHER_EPSILON_DECAY = 0.01  # APF-PID作为专家的概率衰减系数
TEACHER_EPSILON_MIN = 0.05    # 专家概率下限

# ==================== 导航常量 ====================
LASER_MAX_RANGE = 8.0        # 激光雷达有效最大距离(m)
LASER_FRONT_HALF_DEG = 180   # 前方扫描扇区角度(度),仅用于碰撞检测
COLLISION_DISTANCE = 0.6     # 碰撞判定阈值(m)
ARRIVE_DISTANCE = 1.5        # 到达目标判定阈值(m)
DEFAULT_SPEED = 1.0          # 默认速度(m/s)
OBSTACLE_SLOW_RANGE = 4.0    # 进入此范围开始减速(m)
TARGET_SLOW_RANGE = 3.0      # 接近目标时减速阈值(m)
ANGULAR_VELOCITY_MAX = 100   # 策略/奖励/预测使用的内部最大角速度(°/s)
ACTION_TO_SPEED_CONTINOUS_SCALE = 200 # 连续动作映射到速度缩放因子
CONTROL_DT = 0.1             # 控制周期(s),用于连续动作
PPO_TARGET_HEADING_MAX_OFFSET = 120.0  # PPO目标航向最大偏转角(度)
ENABLE_APF_DEBUG_VIEW = True         # 是否打开APF方向实时调试窗口
APF_DEBUG_WINDOW_NAME = "APF Heading Debug"
ROTATE_CONTROL_MODE = "PPO"  # 可选: "PPO" | "PID"
HEADING_PID_KP = 1.2
HEADING_PID_KI = 0.02
HEADING_PID_KD = 0.15
HEADING_PID_INTEGRAL_LIMIT = 60.0

# ==================== 奖励权重 ====================
REWARD_ARRIVE_BONUS = 1000      # 到达奖励
REWARD_COLLISION_PENALTY = -500 # 碰撞惩罚
REWARD_OBSTACLE_PENALTY = -5    # 接近障碍物惩罚
REWARD_NEAR_TARGET_BONUS = 1    # 靠近目标奖励
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
    ppo_target_heading: float
    pid_target_heading: float
    selected_target_heading: float
    breakdown: RewardBreakdown


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
        self.current_episode_step = 0
        self.tb_log_dir = build_tensorboard_log_dir(Path.cwd() / "runs")
        self.tb_writer = TensorBoardMetricsWriter(log_dir=self.tb_log_dir)
        self.enable_episode_logging = False
        self.episode_log_dir = Path.cwd() / "logs"
        self.episode_logger = None

        self.training_logger = TrainingLogger(root_dir="Results")

        self.last_distance = None
        self.enable_apf_debug_view = ENABLE_APF_DEBUG_VIEW

        # PPO智能体
        self.ppo_agent = PPO(
            N_STATES, N_ACTIONS, LR_ACTOR, LR_CRITIC,
            GAMMA, K_EPOCHS, EPS_CLIP, HAS_CONTINUOUS_ACTION, ACTION_STD_INIT,
            writer=self.tb_writer.writer
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
        self.rotate_control_mode = ROTATE_CONTROL_MODE.upper()
        self.teacher_prob = 1.0
        self.teacher_last_selected = False
        self.heading_pid = PID(
            PIDConfig(
                kp=HEADING_PID_KP,
                ki=HEADING_PID_KI,
                kd=HEADING_PID_KD,
                output_limit=ANGULAR_VELOCITY_MAX,
                integral_limit=HEADING_PID_INTEGRAL_LIMIT,
            )
        )
        self.last_heading_debug = {
            "ppo_target_heading": 0.0,
            "pid_target_heading": 0.0,
            "apf_heading_diff": 0.0,
            "ppo_heading_diff": 0.0,
            "selected_target_heading": 0.0,
        }

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

                    try:
                        LogUtil.info(f"第 {epoch} 轮训练开始")
                        self._start_episode_logging(epoch)
                        last_update_metrics = None
                        self.teacher_prob = self._calc_teacher_probability(epoch)

                        # 加载航线
                        self.route = self.ros_ctrl.getRoute()
                        if len(self.route.points) == 0:
                            LogUtil.error("航线数据为空")
                            time.sleep(0.1)
                            continue

                        LogUtil.info(f"航线加载完成: {self.route}")

                        # 初始化本轮状态
                        self._reset_episode_state()
                        self.global_data.route = self.route
                        self.__reloadNavigationRoute(self.route)
                        if not self._reset_and_prepare_episode():
                            LogUtil.error("多次复位后仍处于碰撞初始状态，跳过当前episode")
                            continue

                        # 本轮导航循环
                        self.episode_start_time = time.time()
                        self.episode_step_count = 0
                        self.timeout = False
                        for step in range(MAX_STEP_PER_EPISODE):
                            self.current_episode_step = step + 1
                            self.episode_step_count = self.current_episode_step
                            if self.global_data.device_data.task_status == 0:
                                LogUtil.info(f"步骤 {step} 停止训练")
                                break

                            if (time.time() - self.episode_start_time) > MAX_EPISODE_TIME:
                                LogUtil.info("本轮超时,提前结束")
                                self.timeout = True
                                break

                            self.done = self.navigationHandler(self.next_state, epoch, step)

                            # 定期更新PPO
                            if step > 0 and step % UPDATE_INTERVAL == 0 and self._should_update_ppo():
                                last_update_metrics = self.ppo_agent.update()
                                self.training_logger.log_update(
                                    epoch * MAX_STEP_PER_EPISODE + step,
                                    last_update_metrics,
                                )

                            self.setMonitorParameterValue()

                            if self.done or self.arrive:
                                self._log_episode_metrics(epoch)
                                break

                            time.sleep(0.1)

                        if not self.done and not self.arrive and self.current_episode_step > 0:
                            self._log_episode_metrics(epoch)

                        if self._should_update_ppo():
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
                    finally:
                        self._close_episode_logging()

            except Exception as e:
                LogUtil.error(e)
            finally:
                self._close_episode_logging()
                self._close_tensorboard_writer()
                self.training_logger.close()
                time.sleep(2)

    def _reset_episode_state(self):
        """重置单轮训练的状态变量。"""
        self.episode_reward_sum = 0.0
        self.current_episode_step = 0
        self.next_state = None
        self.last_distance = None
        self.done = False
        self.arrive = False
        self.timeout = False
        self.destPointIndex = -1
        self.destPoint = None
        self.max_distance = 0.0
        self.episode_step_count = 0
        self.last_heading_debug = {
            "ppo_target_heading": 0.0,
            "pid_target_heading": 0.0,
            "apf_heading_diff": 0.0,
            "ppo_heading_diff": 0.0,
            "selected_target_heading": 0.0,
        }
        self.heading_pid.reset()

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

    def _extract_obstacle_feature(self, scan_range: list) -> tuple[float, float]:
        """提取障碍物特征。"""
        if not scan_range:
            return LASER_MAX_RANGE, 0.0

        obstacle_idx = int(np.argmin(scan_range))
        obstacle_min_range = round(float(scan_range[obstacle_idx]), 2)
        return obstacle_min_range, self._scan_feature_index_to_relative_angle(obstacle_idx)

    @staticmethod
    def _scan_feature_index_to_relative_angle(index: int | float) -> float:
        """将重排后的前向雷达特征索引还原为船体系相对角度。"""
        return float(index) * 2.0 - 90.0

    @staticmethod
    def _relative_angle_to_scan_feature_index(angle_deg: float, scan_len: int) -> int | None:
        if scan_len <= 0:
            return None
        if angle_deg < -90.0 or angle_deg >= 90.0:
            return None
        raw_index = int(round((angle_deg + 90.0) / 2.0))
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

    @staticmethod
    def _clip(value: float, limit: float) -> float:
        return max(-limit, min(limit, value))

    @staticmethod
    def _clip_unit(value: float) -> float:
        return max(-1.0, min(1.0, float(value)))

    @staticmethod
    def _calc_teacher_probability(episode: int) -> float:
        epsilon = math.exp(-TEACHER_EPSILON_DECAY * max(0, episode))
        return max(TEACHER_EPSILON_MIN, min(1.0, epsilon))

    def _should_update_ppo(self) -> bool:
        return len(self.ppo_agent.buffer.rewards) >= MIN_BUFFER_SIZE_FOR_UPDATE

    def _reset_and_prepare_episode(self) -> bool:
        for attempt in range(MAX_RESET_RETRIES):
            self.ros_ctrl.reset_unity()
            time.sleep(0.1)
            while self.global_data.device_data.reset_status != 2:
                time.sleep(0.1)

            time.sleep(RESET_SETTLE_DELAY)
            self.ros_ctrl.set_auto_work()

            laser_scan = self._wait_for_laser_data()
            if laser_scan is None:
                continue
            pose_info = self._load_vehicle_pose_info()
            nav_context = self._update_navigation_target(pose_info)
            if nav_context is None:
                continue
            initial_state = self.getState(
                laser_scan,
                pose_info[4],
                nav_context["shipToNextWPDistance"],
                nav_context["degreeAship"],
                pose_info[5],
                pose_info[6],
            )
            obstacle_min_range = initial_state[-2]
            if obstacle_min_range > COLLISION_DISTANCE:
                self.next_state = initial_state
                self.done = False
                return True
            LogUtil.info(
                f"复位后初始障碍过近，跳过本次起点: obstacle_min_range={obstacle_min_range:.2f}, "
                f"attempt={attempt + 1}/{MAX_RESET_RETRIES}"
            )
        return False

    def _compute_apf_target_heading(
        self,
        heading: float,
        target_heading_world: float,
        current_distance: float,
        obstacle_min_range: float,
        obstacle_angle: float,
        angle_diff: float,
    ) -> tuple[float, float]:
        heading_world = self._normalize_heading_360(heading)
        target_world = self._normalize_heading_360(target_heading_world)
        apf_heading_diff = calc_apf_heading_diff(
            angle_diff=angle_diff,
            current_distance=current_distance,
            obstacle_min_range=obstacle_min_range,
            obstacle_angle=obstacle_angle,
            heading_world=heading_world,
            target_heading_world=target_world,
            config=self.reward_config,
        )
        apf_target_heading = self._normalize_signed_angle_diff(heading + apf_heading_diff)
        return apf_target_heading, apf_heading_diff

    def _compute_ppo_target_heading(
        self,
        heading: float,
        turn_ratio: float,
        target_heading_world: float,
        current_distance: float,
        obstacle_min_range: float,
        obstacle_angle: float,
        angle_diff: float,
    ) -> tuple[float, float]:
        heading_world = self._normalize_heading_360(heading)
        apf_heading_diff = calc_apf_heading_diff(
            angle_diff=angle_diff,
            current_distance=current_distance,
            obstacle_min_range=obstacle_min_range,
            obstacle_angle=obstacle_angle,
            heading_world=heading_world,
            target_heading_world=self._normalize_heading_360(target_heading_world),
            config=self.reward_config,
        )
        ppo_heading_diff = self._clip(turn_ratio, 1.0) * PPO_TARGET_HEADING_MAX_OFFSET
        ppo_target_heading = self._normalize_signed_angle_diff(heading + ppo_heading_diff)
        return ppo_target_heading, ppo_heading_diff

    def _pid_output_to_rudder_percent(self, pid_output: float) -> float:
        return round(self._clip(pid_output, ANGULAR_VELOCITY_MAX), 0)

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

        # 动作到控制量的映射
        adviseRotate, adviseSpeed = self._action_to_control(
            action,
            obstacle_min_range,
            current_distance,
            heading,
            degreeAship,
            state,
        )

        # 先把控制量写回环境，再等待环境刷新出新观测
        self.global_data.updateThrottleRudderOutput(
            adviseSpeed, adviseRotate, heading, self.destPointIndex, current_distance
        )

        refreshed_laser_scan = self._wait_for_laser_data()
        if refreshed_laser_scan is None:
            refreshed_laser_scan = laser_scan

        refreshed_pose_info = self._load_vehicle_pose_info()
        refreshed_nav_context = self._update_navigation_target(refreshed_pose_info)
        if refreshed_nav_context is None:
            refreshed_heading = heading
            refreshed_distance = shipToNextWPDistance
            refreshed_degreeAship = degreeAship
            refreshed_speed = pose.speed
            refreshed_rotate_speed = pose.rotate_speed
        else:
            refreshed_heading = refreshed_pose_info[4]
            refreshed_speed = refreshed_pose_info[5]
            refreshed_rotate_speed = refreshed_pose_info[6]
            refreshed_distance = refreshed_nav_context["shipToNextWPDistance"]
            refreshed_degreeAship = refreshed_nav_context["degreeAship"]

        # 获取新状态
        new_state = self.getState(
            refreshed_laser_scan,
            refreshed_heading,
            refreshed_distance,
            refreshed_degreeAship,
            refreshed_speed,
            refreshed_rotate_speed,
        )

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
            episode_elapsed_time=time.time() - self.episode_start_time,
            heading_world=refreshed_heading,
            target_heading_world=refreshed_degreeAship,
            config=self.reward_config,
        )
        reward = breakdown.total_reward
        self._update_apf_debug_view(
            prev_state=state,
            new_state=new_state,
            heading=refreshed_heading,
            target_heading_world=refreshed_degreeAship,
            action=action,
            reward=reward,
        )
        
        if self.arrive:
            LogUtil.info("到达目标!")
        elif self.done:
            LogUtil.info("发生碰撞!")

        return StepResult(
            next_state=np.asarray(new_state),
            reward=reward,
            current_distance=refreshed_distance,
            advise_speed=adviseSpeed,
            advise_rotate=adviseRotate,
            advised_heading=heading,
            distance=shipToNextWPDistance,
            degree_aship=degreeAship,
            obstacle_min_range=state[-2],
            obstacle_angle=state[-1],
            ppo_target_heading=self.last_heading_debug["ppo_target_heading"],
            pid_target_heading=self.last_heading_debug["pid_target_heading"],
            selected_target_heading=self.last_heading_debug["selected_target_heading"],
            breakdown=breakdown,
        )

    def _action_to_control(self, action: np.ndarray, obstacle_min_range: float,
                           current_distance: float, heading: float,
                           target_heading_world: float, state: list) -> tuple:
        if HAS_CONTINUOUS_ACTION:
            """根据模式选择目标航向,再统一交给PID输出舵量。"""
            action_vec = np.asarray(action, dtype=np.float32).reshape(-1)
            if action_vec.size < 2:
                raise ValueError(f"continuous action must have at least 2 elements, got shape={np.asarray(action).shape}")
            turn_ratio = self._clip_unit(action_vec[0])
            speed_ratio = self._clip_unit(action_vec[1])
            apf_target_heading, apf_heading_diff = self._compute_apf_target_heading(
                heading=heading,
                target_heading_world=target_heading_world,
                current_distance=current_distance,
                obstacle_min_range=obstacle_min_range,
                obstacle_angle=state[-1],
                angle_diff=state[-4],
            )
            ppo_target_heading, ppo_heading_diff = self._compute_ppo_target_heading(
                heading=heading,
                turn_ratio=turn_ratio,
                target_heading_world=target_heading_world,
                current_distance=current_distance,
                obstacle_min_range=obstacle_min_range,
                obstacle_angle=state[-1],
                angle_diff=state[-4],
            )
            use_teacher = False
            if MIXED_ROTATE_CONTROL:
                use_teacher = np.random.rand() < self.teacher_prob
            elif self.rotate_control_mode == "PID":
                use_teacher = True
            self.teacher_last_selected = use_teacher
            selected_target_heading = apf_target_heading if use_teacher else ppo_target_heading
            heading_error = self._normalize_signed_angle_diff(selected_target_heading - heading)
            adviseRotate = self._pid_output_to_rudder_percent(self.heading_pid.update(heading_error, CONTROL_DT))
            self.last_heading_debug = {
                "ppo_target_heading": float(ppo_target_heading),
                "pid_target_heading": float(apf_target_heading),
                "apf_heading_diff": float(apf_heading_diff),
                "ppo_heading_diff": float(ppo_heading_diff),
                "selected_target_heading": float(selected_target_heading),
            }

            # 自适应速度
            adviseSpeed = speed_ratio * ACTION_TO_SPEED_CONTINOUS_SCALE

            if obstacle_min_range < 1.0:
                adviseSpeed = min(adviseSpeed, ACTION_TO_SPEED_CONTINOUS_SCALE * 0.2)
            elif obstacle_min_range < 2.0:
                adviseSpeed = min(adviseSpeed, ACTION_TO_SPEED_CONTINOUS_SCALE * 0.4)

            if current_distance < TARGET_SLOW_RANGE:
                adviseSpeed = min(adviseSpeed, ACTION_TO_SPEED_CONTINOUS_SCALE * 0.3)

        else:
            """将离散动作映射为目标航向后再交给PID。"""
            ang_vel = ((self.action_size - 1) / 2 - action) * ANGULAR_VELOCITY_MAX / ((self.action_size - 1) / 2)
            # 自适应速度
            adviseSpeed = DEFAULT_SPEED
            if obstacle_min_range < OBSTACLE_SLOW_RANGE:
                adviseSpeed = 0.1
            if current_distance < TARGET_SLOW_RANGE:
                adviseSpeed = 0.1
            adviseSpeed = min(round(adviseSpeed * 100 / DEFAULT_SPEED, 0), 100)
            apf_target_heading, apf_heading_diff = self._compute_apf_target_heading(
                heading=heading,
                target_heading_world=target_heading_world,
                current_distance=current_distance,
                obstacle_min_range=obstacle_min_range,
                obstacle_angle=state[-1],
                angle_diff=state[-4],
            )
            ppo_target_heading, ppo_heading_diff = self._compute_ppo_target_heading(
                heading=heading,
                turn_ratio=self._clip_unit(ang_vel / ANGULAR_VELOCITY_MAX),
                target_heading_world=target_heading_world,
                current_distance=current_distance,
                obstacle_min_range=obstacle_min_range,
                obstacle_angle=state[-1],
                angle_diff=state[-4],
            )
            use_teacher = self.rotate_control_mode == "PID"
            self.teacher_last_selected = use_teacher
            selected_target_heading = apf_target_heading if use_teacher else ppo_target_heading
            heading_error = self._normalize_signed_angle_diff(selected_target_heading - heading)
            adviseRotate = self._pid_output_to_rudder_percent(self.heading_pid.update(heading_error, CONTROL_DT))
            self.last_heading_debug = {
                "ppo_target_heading": float(ppo_target_heading),
                "pid_target_heading": float(apf_target_heading),
                "apf_heading_diff": float(apf_heading_diff),
                "ppo_heading_diff": float(ppo_heading_diff),
                "selected_target_heading": float(selected_target_heading),
            }

        return adviseRotate, adviseSpeed

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
                curr_apf_heading_diff - turn_action * 100 * CONTROL_DT
            )

            canvas = np.full((720, 760, 3), 248, dtype=np.uint8)
            center = (380, 380)
            cv2.circle(canvas, center, 230, (225, 225, 225), 1)
            cv2.circle(canvas, center, 4, (30, 30, 30), -1)

            self._draw_debug_arrow(canvas, center, 130, heading_world, (50, 50, 50), "ship")
            self._draw_debug_arrow(canvas, center, 180, target_world, (40, 160, 40), "target")
            self._draw_debug_arrow(canvas, center, 195, self._normalize_heading_360(self.last_heading_debug["ppo_target_heading"]), (180, 120, 40), "ppo_pred")
            self._draw_debug_arrow(canvas, center, 215, self._normalize_heading_360(self.last_heading_debug["pid_target_heading"]), (0, 150, 150), "apf_pred")
            self._draw_debug_arrow(canvas, center, 235, self._normalize_heading_360(self.last_heading_debug["selected_target_heading"]), (120, 60, 180), "selected")
            self._draw_debug_arrow(canvas, center, 220, heading_world + curr_apf_heading_diff, (30, 80, 220), "apf_now")
            self._draw_debug_arrow(canvas, center, 150, heading_world + predicted_apf_heading_diff, (180, 60, 180), "apf_next")

            laser_count = max(0, len(new_state) - 6)
            laser_points = np.asarray(new_state[:laser_count], dtype=np.float32)
            if laser_points.size > 0:
                obstacle_idx = int(np.argmin(laser_points))
                for idx, beam_range in enumerate(laser_points):
                    # 源代码中的前方180°扇区按2°采样，这里按索引还原角度。
                    relative_deg = self._scan_feature_index_to_relative_angle(idx)
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

                obstacle_relative_deg = float(new_state[-1])
                obstacle_angle_idx = self._relative_angle_to_scan_feature_index(
                    obstacle_relative_deg,
                    laser_points.size,
                )
                if obstacle_angle_idx is not None and 0 <= obstacle_angle_idx < laser_points.size:
                    obstacle_point_range = float(laser_points[obstacle_angle_idx])
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
                        "nearest_lidar_obstacle",
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
                f"obstacle_angle(relative): {new_state[-1]:.2f} deg",
                f"obstacle_feature_index: {obstacle_angle_idx}",
                f"obstacle_min_range: {new_state[-2]:.2f} m",
                f"prev_apf_heading_diff: {prev_apf_heading_diff:.2f} deg",
                f"curr_apf_heading_diff: {curr_apf_heading_diff:.2f} deg",
                f"pred_apf_heading_diff: {predicted_apf_heading_diff:.2f} deg",
                f"teacher_prob: {self.teacher_prob:.3f}",
                f"teacher_selected: {int(self.teacher_last_selected)}",
                f"laser_points: {laser_count}",
                f"turn_action: {turn_action:.3f}",
                f"ppo_heading_diff: {self.last_heading_debug['ppo_heading_diff']:.2f}",
                f"selected_target: {self.last_heading_debug['selected_target_heading']:.2f}",
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

            self._log_episode_snapshot(pose_info, nav_context, laser_scan)

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

    def _log_episode_snapshot(self, pose_info: tuple, nav_context: dict, laser_scan: Any):
        if not self.enable_episode_logging or self.episode_logger is None:
            return
        obstacle = self._build_obstacle_snapshot(laser_scan, pose_info[4])
        target = self._point_to_dict(self.destPoint)
        ship = {
            "lng": pose_info[2],
            "lat": pose_info[3],
        }
        laser = list(getattr(laser_scan, "ranges", []) or [])
        self.episode_logger.log_snapshot(
            obstacle=obstacle,
            target=target,
            ship=ship,
            laser=laser,
        )

    def _start_episode_logging(self, episode: int):
        if not self.enable_episode_logging:
            return
        if self.episode_logger is None:
            self.episode_logger = EpisodeDataLogger(self.episode_log_dir)
        self.episode_logger.start_episode(episode)

    def _close_episode_logging(self):
        if self.episode_logger is None:
            return
        self.episode_logger.close()

    def _build_obstacle_snapshot(self, laser_scan: Any, heading: float) -> dict:
        ranges = list(getattr(laser_scan, "ranges", []) or [])
        if not ranges:
            return {"lng": None, "lat": None, "distance": None, "angle": None}

        valid_points = []
        for index, value in enumerate(ranges):
            if value in (None, float("Inf")):
                continue
            if isinstance(value, float) and np.isnan(value):
                continue
            if value <= 0 or value > LASER_MAX_RANGE:
                continue
            valid_points.append((index, float(value)))

        if not valid_points:
            return {"lng": None, "lat": None, "distance": None, "angle": None}

        min_index, min_distance = min(valid_points, key=lambda item: item[1])
        heading_rad = math.radians(heading)
        angle_increment = self._safe_float(getattr(laser_scan, "angle_increment", 0.0))
        angle_min = self._safe_float(getattr(laser_scan, "angle_min", 0.0))
        relative_angle = angle_min + min_index * angle_increment
        world_angle = heading_rad + relative_angle
        relative_angle_deg = math.degrees(relative_angle)

        return {
            "lng": round(min_distance * math.cos(world_angle), 6),
            "lat": round(min_distance * math.sin(world_angle), 6),
            "distance": round(min_distance, 6),
            "angle": round(relative_angle_deg, 6),
        }

    @staticmethod
    def _point_to_dict(point: Any) -> dict:
        if point is None:
            return {"lng": None, "lat": None}
        return {
            "lng": getattr(point, "lng", None),
            "lat": getattr(point, "lat", None),
        }

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

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

        if self.tb_writer is not None:
            if self.rotate_control_mode == "PPO":
                self.tb_writer.add_scalar("heading/ppo_target", result.ppo_target_heading, self.current_episode_step)
            self.tb_writer.add_scalar("heading/apf_target", result.pid_target_heading, self.current_episode_step)
            self.tb_writer.add_scalar("heading/selected_target", result.selected_target_heading, self.current_episode_step)
            self.tb_writer.flush()

        # 输出控制量（修复点：补全参数）
        self._output_control_commands(result, episode, step, nav_context['nextPointIndex'])

        if self.rotate_control_mode == "PID":
            LogUtil.debug(
                f"Step={step} Action={action} Reward={result.reward:.2f} "
                f"Rudder={result.advise_rotate:.1f} "
                f"Done={self.done} Arrive={self.arrive}"
            )
        else:
            LogUtil.debug(
                f"Step={step} Action={action} Reward={result.reward:.2f} "
                f"Rudder={result.advise_rotate:.1f} "
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

    def _log_navigation_result(self, nav_context: dict):
        """记录导航结果日志。"""
        LogUtil.info(
            f"航点={self.destPointIndex} 距离={nav_context['shipToNextWPDistance']:.1f}m "
            f"速度=... 转向=..."
        )

    def _log_episode_metrics(self, episode: int):
        if self.tb_writer is None:
            return
        self.tb_writer.log_episode(
            episode=episode,
            reward=self.episode_reward_sum,
            length=self.current_episode_step,
        )

    def _close_tensorboard_writer(self):
        if self.tb_writer is None:
            return
        try:
            self.tb_writer.close()
        except Exception as exc:
            LogUtil.error(f"关闭TensorBoard writer失败: {exc}")
        finally:
            self.tb_writer = None

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
    
    def close(self):
        if getattr(self, "enable_apf_debug_view", False):
            cv2.destroyAllWindows()
        if hasattr(self, "training_logger") and self.training_logger is not None:
            self.training_logger.close()


# 向后兼容：保留旧类名供main.py等外部引用
DQN_NAV = PPONav
