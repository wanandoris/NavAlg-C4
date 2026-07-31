import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class RewardConfig:
    reward_arrive_bonus: float = 80
    reward_collision_penalty: float = -50
    reward_near_target_bonus: float = 1
    reward_weight_distance: float = 2.4
    reward_weight_obstacle: float = 0.9
    reward_weight_heading: float = 1.0
    reward_weight_time: float = 0.2
    progress_scale: float = 14.0
    progress_reward_weight: float = 0.75
    proximity_reward_weight: float = 0.25
    proximity_exponent: float = 2.2
    proximity_normalizer: float = 12.0
    step_penalty: float = -0.005
    max_episode_time: float = 300.0
    min_arrive_time_weight: float = 0.35
    target_slow_range: float = 3.0
    angular_velocity_max: float = 100
    control_dt: float = 0.1
    apf_attractive_gain: float = 1.0
    apf_repulsive_gain: float = 8.0
    apf_obstacle_influence_range: float = 2.5
    time_exponent: float = 1.4


DEFAULT_REWARD_CONFIG = RewardConfig()


@dataclass(frozen=True)
class RewardBreakdown:
    distance_reward: float
    heading_reward: float
    obstacle_reward: float
    time_reward: float
    total_reward: float
    distance_raw: float
    heading_raw: float
    obstacle_raw: float

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

    time_reward = 0.0
    if episode_elapsed_time is not None and config.max_episode_time > 0:
        t_norm = episode_elapsed_time / config.max_episode_time
        time_reward_raw = -math.exp(config.time_exponent * t_norm)
        time_reward = time_reward_raw * config.reward_weight_time

    distance_reward = distance_reward_raw * config.reward_weight_distance
    obstacle_reward = obstacle_reward_raw * config.reward_weight_obstacle
    heading_reward = heading_reward_raw * config.reward_weight_heading

    reward = (
        distance_reward
        + obstacle_reward
        + heading_reward
        + time_reward
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
        time_reward=time_reward,
        total_reward=reward,
        distance_raw=distance_reward_raw,
        heading_raw=heading_reward_raw,
        obstacle_raw=obstacle_reward_raw,
    )


def _extract_turn_action(action: int | float | list | tuple) -> float:
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
        if flat:
            return float(flat[0])
        return 0.0
    return float(action)


def _clip_turn_action(turn_action: float) -> float:
    return max(-1.0, min(1.0, float(turn_action)))


def calc_distance_reward(
    current_distance: float,
    prev_distance: float | None,
    max_distance: float,
    config: RewardConfig = DEFAULT_REWARD_CONFIG,
) -> float:
    progress_term = calc_progress_reward(prev_distance, current_distance, config)
    proximity_term = calc_proximity_reward(
        current_distance=current_distance,
        max_distance=max_distance,
        config=config,
    )
    return (
        config.progress_reward_weight * progress_term
        + config.proximity_reward_weight * proximity_term
    )


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
    return config.min_arrive_time_weight + (1.0 - config.min_arrive_time_weight) * math.exp(-4 * progress)


def calc_progress_reward(
    prev_distance: float | None,
    current_distance: float,
    config: RewardConfig = DEFAULT_REWARD_CONFIG,
) -> float:
    if prev_distance is None or prev_distance <= 0:
        return 0.0
    progress = prev_distance - current_distance
    normalized_progress = progress / prev_distance
    clipped_progress = max(-1.0, min(1.0, normalized_progress))
    return clipped_progress * config.progress_scale


def calc_proximity_reward(
    current_distance: float,
    max_distance: float,
    config: RewardConfig = DEFAULT_REWARD_CONFIG,
) -> float:
    if current_distance <= 0:
        return 1.0

    distance_scale = max(float(max_distance), config.proximity_normalizer, 1e-6)
    normalized_distance = max(0.0, current_distance) / distance_scale
    proximity = math.exp(-config.proximity_exponent * normalized_distance)
    return max(0.0, min(1.0, proximity))


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
    turn_action = _extract_turn_action(action)
    turn_action = _clip_turn_action(turn_action)

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
    obstacle_points: Sequence[tuple[float, float]] | None = None,
) -> float:
    heading_world = 0.0 if heading_world is None else heading_world
    if target_heading_world is None:
        target_heading_world = heading_world + angle_diff

    target_rad = math.radians(target_heading_world)
    attractive_strength = config.apf_attractive_gain * max(float(current_distance), 0.0)
    attractive_x = attractive_strength * math.cos(target_rad)
    attractive_y = attractive_strength * math.sin(target_rad)

    repulsive_x = 0.0
    repulsive_y = 0.0
    rho_0 = config.apf_obstacle_influence_range
    points = obstacle_points or [(obstacle_min_range, obstacle_angle)]
    for point_range, point_angle in points:
        rho = max(float(point_range), 1e-3)
        if rho >= rho_0:
            continue
        force_magnitude = config.apf_repulsive_gain * (1.0 / rho - 1.0 / rho_0) / (rho * rho)
        obstacle_relative_deg = float(point_angle)
        obstacle_world_deg = heading_world + obstacle_relative_deg
        obstacle_rad = math.radians(obstacle_world_deg)
        repulsive_x += -force_magnitude * math.cos(obstacle_rad)
        repulsive_y += -force_magnitude * math.sin(obstacle_rad)

    apf_x = attractive_x + repulsive_x
    apf_y = attractive_y + repulsive_y
    if abs(apf_x) < 1e-6 and abs(apf_y) < 1e-6:
        return _normalize_signed_angle_diff(angle_diff)
    apf_heading_world = math.degrees(math.atan2(apf_y, apf_x))
    return _normalize_signed_angle_diff(apf_heading_world - heading_world)


def _normalize_signed_angle_diff(angle: float) -> float:
    return (angle + 180) % 360 - 180
