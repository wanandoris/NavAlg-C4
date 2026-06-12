import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    reward_arrive_bonus: float = 1000
    reward_collision_penalty: float = -500
    reward_obstacle_penalty: float = -5
    reward_near_target_bonus: float = 1
    reward_weight_distance: float = 0.6
    reward_weight_obstacle: float = 0.2
    reward_weight_heading: float = 0.2
    target_slow_range: float = 3.0
    angular_velocity_max: float = 100
    control_dt: float = 0.1
    has_continuous_action: bool = True
    n_actions: int = 1
    speed_scale: float = 100.0


DEFAULT_REWARD_CONFIG = RewardConfig()


def compute_reward(
    state: list,
    action: int | float | list | tuple,
    max_distance: float,
    angle_diff: float,
    arrive: bool,
    done: bool,
    config: RewardConfig = DEFAULT_REWARD_CONFIG,
) -> float:
    obstacle_min_range = state[-2]
    current_distance = state[-3]
    distance_reward = calc_distance_reward(current_distance, max_distance)
    heading_reward = calc_heading_reward(
        action=action,
        angle_diff=angle_diff if angle_diff is not None else state[-4],
        current_distance=current_distance,
        max_distance=max_distance,
        config=config,
    )

    obstacle_reward = 0.0
    if obstacle_min_range < 3:
        obstacle_reward = config.reward_obstacle_penalty
    elif current_distance < config.target_slow_range:
        obstacle_reward = config.reward_near_target_bonus

    reward = (
        distance_reward * config.reward_weight_distance
        + obstacle_reward * config.reward_weight_obstacle
        + heading_reward * config.reward_weight_heading
    )

    if arrive:
        reward += config.reward_arrive_bonus
    elif done:
        reward += config.reward_collision_penalty

    return reward


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


def calc_distance_reward(current_distance: float, max_distance: float) -> float:
    if current_distance <= 1 or max_distance <= 0:
        return 0.0
    reward = 1 - (current_distance / max_distance)
    return reward * 2 if reward < 0 else reward * 5


def calc_heading_reward(
    action: int | float | list | tuple,
    angle_diff: float,
    current_distance: float,
    max_distance: float,
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

    predicted_angle_diff = _normalize_signed_angle_diff(
        angle_diff - turn_action * config.angular_velocity_max * config.control_dt
    )
    heading_reward = 1 - 2 * (abs(predicted_angle_diff) / 180.0)
    return round(heading_reward, 2)


def _normalize_signed_angle_diff(angle: float) -> float:
    return (angle + 180) % 360 - 180
