import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    reward_arrive_bonus: float = 1000
    reward_collision_penalty: float = -500
    reward_near_target_bonus: float = 1
    reward_weight_distance: float = 0.8
    reward_weight_obstacle: float = 0.4
    reward_weight_heading: float = 0.4
    progress_scale: float = 10.0
    obstacle_safe_range: float = 3.0
    obstacle_penalty_scale: float = 10.0
    step_penalty: float = -0.01
    max_episode_time: float = 300.0
    min_arrive_time_weight: float = 0.2
    target_slow_range: float = 3.0
    angular_velocity_max: float = 100
    control_dt: float = 0.1
    has_continuous_action: bool = True
    n_actions: int = 1
    speed_scale: float = 100.0
    apf_attractive_gain: float = 1.0
    apf_repulsive_gain: float = 1.0
    apf_obstacle_influence_range: float = 3.0
    apf_heading_repulsive_weight: float = 1.0


DEFAULT_REWARD_CONFIG = RewardConfig()


@dataclass(frozen=True)
class RewardBreakdown:
    distance_reward: float
    heading_reward: float
    obstacle_reward: float
    total_reward: float

    distance_raw: float
    heading_raw: float
    obstacle_raw: float


def compute_reward(
    state: list,
    action: int | float | list | tuple,
    max_distance: float,
    angle_diff: float,
    arrive: bool,
    done: bool,
    prev_state: list | None = None,
    prev_distance: float | None = None,
    heading_world: float | None = None,
    target_heading_world: float | None = None,
    episode_elapsed_time: float | None = None,
    config: RewardConfig = DEFAULT_REWARD_CONFIG,
) -> float:
    return compute_reward_breakdown(
        state=state,
        action=action,
        max_distance=max_distance,
        angle_diff=angle_diff,
        arrive=arrive,
        done=done,
        prev_state=prev_state,
        prev_distance=prev_distance,
        heading_world=heading_world,
        target_heading_world=target_heading_world,
        episode_elapsed_time=episode_elapsed_time,
        config=config,
    ).total_reward


def compute_reward_breakdown(
    state: list,
    action: int | float | list | tuple,
    max_distance: float,
    angle_diff: float,
    arrive: bool,
    done: bool,
    prev_state: list | None = None,
    prev_distance: float | None = None,
    heading_world: float | None = None,
    target_heading_world: float | None = None,
    episode_elapsed_time: float | None = None,
    config: RewardConfig = DEFAULT_REWARD_CONFIG,
) -> RewardBreakdown:
    obstacle_min_range = state[-2]
    obstacle_angle = state[-1]
    current_distance = state[-3]
    distance_reward_raw = calc_progress_reward(prev_distance, current_distance)
    if prev_distance is None:
        distance_reward_raw = calc_distance_reward(
        current_distance=current_distance,
        prev_distance=prev_distance,
        max_distance=max_distance,
        config=config,
    )
    heading_reward_raw = calc_apf_heading_reward(
        action=action,
        angle_diff=angle_diff if angle_diff is not None else state[-4],
        current_distance=current_distance,
        max_distance=max_distance,
        prev_state=prev_state,
        obstacle_min_range=obstacle_min_range,
        obstacle_angle=obstacle_angle,
        heading_world=heading_world,
        target_heading_world=target_heading_world,
        config=config,
    )

    obstacle_reward_raw = -calc_repulsive_potential(obstacle_min_range, config)
    if current_distance < config.target_slow_range:
        obstacle_reward_raw = config.reward_near_target_bonus

    distance_reward = distance_reward_raw * config.reward_weight_distance
    obstacle_reward = obstacle_reward_raw * config.reward_weight_obstacle
    heading_reward = heading_reward_raw * config.reward_weight_heading

    reward = (
        distance_reward
        + obstacle_reward
        + heading_reward
        + config.step_penalty
    )

    if arrive:
        time_reward_weight = calc_time_reward_weight(episode_elapsed_time, config)
        reward += config.reward_arrive_bonus * time_reward_weight
    elif done:
        reward += config.reward_collision_penalty

    return RewardBreakdown(
        distance_reward=distance_reward,
        heading_reward=heading_reward,
        obstacle_reward=obstacle_reward,
        total_reward=reward,
        distance_raw=distance_reward_raw,
        heading_raw=heading_reward_raw,
        obstacle_raw=obstacle_reward_raw,
    )


def _split_action(action: int | float | list | tuple) -> tuple[float, float]:
    """将动作统一拆成(转向, 速度)两维。"""
    if hasattr(action, "tolist"):
        action = action.tolist()
    if isinstance(action, (list, tuple)):
        flat = []
        for item in action:
            if hasattr(item, "tolist"):
                item = item.tolist()
            if isinstance(item, (list, tuple)):
                flat.extend(float(x) for x in item)
            else:
                flat.append(float(item))
        if len(flat) >= 2:
            return float(flat[0]), float(flat[1])
        if len(flat) == 1:
            return float(flat[0]), 0.0
        return 0.0, 0.0
    return float(action), 0.0


def calc_distance_reward(
    current_distance: float,
    prev_distance: float | None,
    max_distance: float,
    config: RewardConfig = DEFAULT_REWARD_CONFIG,
) -> float:
    if current_distance <= 1:
        return 0.0

    if prev_distance is not None:
        return (prev_distance - current_distance) * config.progress_scale

    if max_distance <= 0:
        return 0.0

    reward = 1 - (current_distance / max_distance)
    return reward * 2 if reward < 0 else reward * 5


def calc_obstacle_reward(
    obstacle_min_range: float,
    config: RewardConfig = DEFAULT_REWARD_CONFIG,
) -> float:
    if obstacle_min_range >= config.obstacle_safe_range:
        return 0.0

    danger_ratio = (config.obstacle_safe_range - obstacle_min_range) / config.obstacle_safe_range
    return -(danger_ratio ** 2) * config.obstacle_penalty_scale


def calc_time_reward_weight(
    episode_elapsed_time: float | None,
    config: RewardConfig = DEFAULT_REWARD_CONFIG,
) -> float:
    if episode_elapsed_time is None:
        return 1.0

    if config.max_episode_time <= 0:
        return 1.0

    progress = episode_elapsed_time / config.max_episode_time
    progress = max(0.0, min(1.0, progress))

    return config.min_arrive_time_weight + (
        1.0 - config.min_arrive_time_weight
    ) * math.exp(-4 * progress)


def calc_progress_reward(prev_distance: float | None, current_distance: float) -> float:
    """基于距离变化量的进度奖励。

    前进则为正，远离目标则为负；数值尽量保持小而稳定，避免压过终止奖励。
    """
    if prev_distance is None:
        return 0.0
    if prev_distance <= 0:
        return 0.0
    progress = prev_distance - current_distance
    return max(-1.0, min(1.0, progress / prev_distance))


def calc_heading_reward(
    action: int | float | list | tuple,
    angle_diff: float,
    current_distance: float,
    max_distance: float,
    obstacle_min_range: float | None = None,
    obstacle_angle: float | None = None,
    heading_world: float | None = None,
    target_heading_world: float | None = None,
    config: RewardConfig = DEFAULT_REWARD_CONFIG,
) -> float:
    distance_rate = 2 ** (current_distance / max_distance) if max_distance > 0 else 1.0
    turn_action, _ = _split_action(action)

    if not config.has_continuous_action:
        yaw_rewards = []
        pi = math.pi
        for i in range(config.n_actions):
            angle = -pi / 4 + angle_diff + (pi / 8 * i) + pi / 2
            tr = 1 - 4 * abs(0.5 - math.modf(0.25 + 0.5 * angle % (2 * pi) / pi)[0])
            yaw_rewards.append(tr)
        return round(yaw_rewards[action] * 5, 2) * distance_rate

    if obstacle_min_range is not None and obstacle_angle is not None:
        apf_heading_diff = calc_apf_heading_diff(
            angle_diff=angle_diff,
            current_distance=current_distance,
            obstacle_min_range=obstacle_min_range,
            obstacle_angle=obstacle_angle,
            heading_world=heading_world,
            target_heading_world=target_heading_world,
            config=config,
        )
    else:
        apf_heading_diff = angle_diff

    predicted_angle_diff = _normalize_signed_angle_diff(
        apf_heading_diff - turn_action * config.angular_velocity_max * config.control_dt
    )
    heading_reward = 1 - 2 * (abs(predicted_angle_diff) / 180.0)
    return round(heading_reward, 2) * distance_rate


def calc_repulsive_potential(
    obstacle_min_range: float,
    config: RewardConfig = DEFAULT_REWARD_CONFIG,
) -> float:
    rho = max(float(obstacle_min_range), 1e-3)
    rho_0 = config.apf_obstacle_influence_range
    if rho >= rho_0:
        return 0.0
    return 0.5 * config.apf_repulsive_gain * ((1.0 / rho) - (1.0 / rho_0)) ** 2


def calc_apf_heading_reward(
    action: int | float | list | tuple,
    angle_diff: float,
    current_distance: float,
    max_distance: float,
    prev_state: list | None,
    obstacle_min_range: float,
    obstacle_angle: float,
    heading_world: float | None,
    target_heading_world: float | None,
    config: RewardConfig = DEFAULT_REWARD_CONFIG,
) -> float:
    distance_rate = 2 ** (current_distance / max_distance) if max_distance > 0 else 1.0
    turn_action, _ = _split_action(action)

    if not config.has_continuous_action:
        return calc_heading_reward(
            action=action,
            angle_diff=angle_diff,
            current_distance=current_distance,
            max_distance=max_distance,
            config=config,
        )

    apf_heading_diff = calc_apf_heading_diff(
        angle_diff=angle_diff,
        current_distance=current_distance,
        obstacle_min_range=obstacle_min_range,
        obstacle_angle=obstacle_angle,
        heading_world=heading_world,
        target_heading_world=target_heading_world,
        config=config,
    )
    predicted_heading_diff = _normalize_signed_angle_diff(
        apf_heading_diff - turn_action * config.angular_velocity_max * config.control_dt
    )
    if prev_state is None or len(prev_state) < 4:
        heading_reward = 1 - 2 * (abs(predicted_heading_diff) / 180.0)
        return round(heading_reward, 2) * distance_rate

    prev_angle_diff = prev_state[-4]
    prev_current_distance = prev_state[-3]
    prev_obstacle_min_range = prev_state[-2]
    prev_obstacle_angle = prev_state[-1]
    prev_apf_heading_diff = calc_apf_heading_diff(
        angle_diff=prev_angle_diff,
        current_distance=prev_current_distance,
        obstacle_min_range=prev_obstacle_min_range,
        obstacle_angle=prev_obstacle_angle,
        heading_world=heading_world,
        target_heading_world=target_heading_world,
        config=config,
    )
    prev_align = math.cos(math.radians(prev_apf_heading_diff))
    curr_align = math.cos(math.radians(predicted_heading_diff))
    heading_progress = curr_align - prev_align
    return round(heading_progress * 2.0, 2)


def calc_apf_heading_diff(
    angle_diff: float,
    current_distance: float,
    obstacle_min_range: float,
    obstacle_angle: float,
    heading_world: float | None,
    target_heading_world: float | None,
    config: RewardConfig = DEFAULT_REWARD_CONFIG,
) -> float:
    heading_world = 0.0 if heading_world is None else heading_world
    if target_heading_world is None:
        target_heading_world = heading_world + angle_diff

    # ---------- 引力矢量（与论文公式一致） ----------
    target_rad = math.radians(target_heading_world)
    attractive_strength = config.apf_attractive_gain * max(float(current_distance), 0.0)
    attractive_x = attractive_strength * math.cos(target_rad)
    attractive_y = attractive_strength * math.sin(target_rad)

    # ---------- 斥力矢量（论文梯度公式，而非势能） ----------
    repulsive_x = 0.0
    repulsive_y = 0.0
    rho = max(float(obstacle_min_range), 1e-3)
    rho_0 = config.apf_obstacle_influence_range
    if rho < rho_0:
        # 斥力大小：k * (1/rho - 1/rho_0) * (1/rho^2)
        force_magnitude = config.apf_repulsive_gain * (1.0 / rho - 1.0 / rho_0) / (rho * rho)
        obstacle_relative_deg = float(obstacle_angle)
        obstacle_world_deg = heading_world + obstacle_relative_deg
        obstacle_rad = math.radians(obstacle_world_deg)
        # 斥力方向：从障碍物指向船舶（即与障碍物方向相反）
        # 保留原代码的负号，使船舶远离障碍物
        repulsive_strength = force_magnitude * config.apf_heading_repulsive_weight
        repulsive_x = -repulsive_strength * math.cos(obstacle_rad)
        repulsive_y = -repulsive_strength * math.sin(obstacle_rad)

    # ---------- 合力 ----------
    apf_x = attractive_x + repulsive_x
    apf_y = attractive_y + repulsive_y
    if abs(apf_x) < 1e-6 and abs(apf_y) < 1e-6:
        return _normalize_signed_angle_diff(angle_diff)
    apf_heading_world = math.degrees(math.atan2(apf_y, apf_x))
    return _normalize_signed_angle_diff(apf_heading_world - heading_world)


def _normalize_signed_angle_diff(angle: float) -> float:
    return (angle + 180) % 360 - 180
