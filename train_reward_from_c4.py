"""
Train RewardWeightNet from data_collection.csv

你的 CSV 字段：
    distance, degreeAship, obstacle_min_range, obstacle_angle,
    distance_reward_raw, heading_reward_raw, obstacle_reward_raw,
    task_reward, done

映射关系：
    features = [d_goal, theta_goal, d_obs_min, theta_obs_min]
    reward_components = [r_goal_progress, r_goal_proximity, r_obs_dist, r_apf]

说明：
    1. distance              -> d_goal
    2. degreeAship           -> theta_goal
    3. obstacle_min_range    -> d_obs_min
    4. obstacle_angle        -> theta_obs_min
    5. distance_reward_raw   -> r_goal_progress
    6. distance              -> r_goal_proximity = 1 - d_goal_norm
    7. obstacle_reward_raw   -> r_obs_dist
    8. heading_reward_raw -> r_apf
    9. task_reward           -> task_rewards，用于计算未来回报 G_t
    10. done                 -> dones

注意：
    total_reward 如果是手工加权后的连续奖励，不应该再作为训练目标。
    task_reward 应该是任务级离散奖励，例如到达目标 +1，碰撞 -1，普通步 -0.01。
"""

import os
import csv
import math
import json
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import torch

try:
    from usvlib4ros.user.c4_project import (
        RewardWeightConfig,
        RewardWeightNet,
        pretrain_reward_net,
        train_reward_net_by_correlation,
        save_reward_net,
    )
except ImportError:
    from c4_project import (
        RewardWeightConfig,
        RewardWeightNet,
        pretrain_reward_net,
        train_reward_net_by_correlation,
        save_reward_net,
    )


# ==========================================================
# 1. 路径配置
# ==========================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.dirname(CURRENT_DIR)
CSV_PATH = os.path.join(PACKAGE_DIR, "data_collection.csv")
SAVE_PATH = os.path.join(CURRENT_DIR, "reward_weight_net_from_csv.pth")
NORMALIZER_PATH = os.path.join(CURRENT_DIR, "reward_input_normalizer.json")


# ==========================================================
# 2. 工具函数
# ==========================================================

def to_float(value, default: float = 0.0) -> float:
    """安全转 float，避免 CSV 里有空字符串时报错。"""
    try:
        if value is None or value == "":
            return default
        return float(value)
    except ValueError:
        return default


def min_max_normalize(values: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    把一列数据归一化到 [0, 1]。

    例如 distance 原始值可能是 11.9、12.0、13.0，
    直接送进网络数值偏大，所以建议归一化。
    """
    v_min = values.min()
    v_max = values.max()
    return (values - v_min) / (v_max - v_min + eps)


def min_max_normalize_with_range(
    values: torch.Tensor,
    value_min: float,
    value_max: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """使用固定 min/max 归一化，适合在线实时数据。"""
    return torch.clamp((values - value_min) / (value_max - value_min + eps), 0.0, 1.0)


def angle_degree_to_minus1_1(values: torch.Tensor) -> torch.Tensor:
    """
    把角度制角度归一化到 [-1, 1] 附近。

    如果你的角度范围是 [-180, 180]，除以 180 后正好是 [-1, 1]。
    如果你的角度范围是 [0, 360]，这里会先映射到 [-180, 180]。
    """
    values = torch.remainder(values + 180.0, 360.0) - 180.0
    return values / 180.0


def get_task_reward_from_raw(
    raw_step: Dict[str, float],
    step_penalty: float = -0.01,
    goal_reward: float = 1.0,
    collision_penalty: float = -1.0,
    timeout_penalty: float = -0.3,
) -> float:
    """
    生成用于计算 G_t 的任务级奖励。

    优先级：
        1. 如果数据里已经有 task_reward，直接使用。
        2. 如果有 reach_goal / collision / timeout 标志，按任务结果生成离散奖励。
        3. 如果只有 done 和 total_reward，终止时用 total_reward 的正负粗略判断成功/失败。
        4. 普通步默认给一个很小的步进惩罚。

    注意：
        不再默认用 total_reward 作为训练目标，因为 total_reward 可能已经包含手工权重。
    """
    if raw_step.get("task_reward") not in (None, ""):
        return to_float(raw_step.get("task_reward"))

    reach_goal = to_float(raw_step.get("reach_goal")) > 0.5 or to_float(raw_step.get("reached_goal")) > 0.5
    collision = to_float(raw_step.get("collision")) > 0.5 or to_float(raw_step.get("is_collision")) > 0.5
    timeout = to_float(raw_step.get("timeout")) > 0.5 or to_float(raw_step.get("is_timeout")) > 0.5
    done = to_float(raw_step.get("done")) > 0.5

    if reach_goal:
        return goal_reward
    if collision:
        return collision_penalty
    if timeout:
        return timeout_penalty

    if done:
        total_reward = to_float(raw_step.get("total_reward"))
        return goal_reward if total_reward > 0.0 else collision_penalty

    return step_penalty


@dataclass
class RewardInputNormalizer:
    """
    保存离线训练时的归一化范围，实时更新时必须复用同一套范围。

    不能在实时单步数据上重新做 min-max，否则网络输入尺度会漂移。
    """

    distance_min: float
    distance_max: float
    obstacle_min_range_min: float
    obstacle_min_range_max: float

    @classmethod
    def from_raw_tensors(
        cls,
        distance: torch.Tensor,
        obstacle_min_range: torch.Tensor,
    ) -> "RewardInputNormalizer":
        return cls(
            distance_min=float(distance.min().item()),
            distance_max=float(distance.max().item()),
            obstacle_min_range_min=float(obstacle_min_range.min().item()),
            obstacle_min_range_max=float(obstacle_min_range.max().item()),
        )

    @classmethod
    def load(cls, path: str = NORMALIZER_PATH) -> "RewardInputNormalizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def save(self, path: str = NORMALIZER_PATH) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)

    def normalize_distance(self, value) -> torch.Tensor:
        value = torch.as_tensor(value, dtype=torch.float32)
        return min_max_normalize_with_range(value, self.distance_min, self.distance_max)

    def normalize_obstacle_min_range(self, value) -> torch.Tensor:
        value = torch.as_tensor(value, dtype=torch.float32)
        return min_max_normalize_with_range(
            value,
            self.obstacle_min_range_min,
            self.obstacle_min_range_max,
        )


def raw_step_to_network_input(
    raw_step: Dict[str, float],
    normalizer: RewardInputNormalizer,
) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
    """
    把对方实时传来的一步原始数据，转换成 c4_project.py 在线更新需要的数据。

    输入 raw_step 字段与 data_collection.csv 保持一致：
        distance, degreeAship, obstacle_min_range, obstacle_angle,
        distance_reward_raw, heading_reward_raw, obstacle_reward_raw,
        task_reward, done

    返回：
        features: [4] = [d_goal_norm, theta_goal_norm, d_obs_min_norm, theta_obs_norm]
        reward_components: [4] = [r_goal_progress, r_goal_proximity, r_obs_dist, r_apf]
        task_reward: float
        done: float
    """
    distance = to_float(raw_step.get("distance"))
    degree_aship = to_float(raw_step.get("degreeAship"))
    obstacle_min_range = to_float(raw_step.get("obstacle_min_range"))
    obstacle_angle = to_float(raw_step.get("obstacle_angle"))

    distance_reward_raw = to_float(raw_step.get("distance_reward_raw"))
    heading_reward_raw = to_float(raw_step.get("heading_reward_raw"))
    obstacle_reward_raw = to_float(raw_step.get("obstacle_reward_raw"))
    task_reward = get_task_reward_from_raw(raw_step)
    done = to_float(raw_step.get("done"))

    d_goal = normalizer.normalize_distance(distance)
    theta_goal = angle_degree_to_minus1_1(torch.tensor(degree_aship, dtype=torch.float32))
    d_obs_min = normalizer.normalize_obstacle_min_range(obstacle_min_range)
    theta_obs_min = angle_degree_to_minus1_1(torch.tensor(obstacle_angle, dtype=torch.float32))

    features = torch.stack([d_goal, theta_goal, d_obs_min, theta_obs_min]).float()
    reward_components = torch.tensor(
        [
            distance_reward_raw,
            1.0 - float(d_goal.item()),
            obstacle_reward_raw,
            heading_reward_raw,
        ],
        dtype=torch.float32,
    )

    return features, reward_components, task_reward, done


def load_csv_dataset(
    csv_path: str,
    normalizer: RewardInputNormalizer = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, RewardInputNormalizer]:
    """
    读取 data_collection.csv，并转换为训练需要的四个 Tensor。

    返回：
        features: [N, 4]
            [d_goal, theta_goal, d_obs_min, theta_obs_min]

        reward_components: [N, 4]
            [r_goal_progress, r_goal_proximity, r_obs_dist, r_apf]

        task_rewards: [N]
            优先用 task_reward 作为任务奖励，用来计算未来回报 G_t。
            如果 CSV 没有 task_reward，则根据 reach_goal / collision / done 自动生成。

        dones: [N]
            episode 是否结束。

        normalizer:
            离线训练得到的固定归一化参数，实时更新时继续复用。
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到 CSV 文件: {csv_path}")

    rows: List[Dict[str, str]] = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if len(rows) == 0:
        raise ValueError("CSV 文件为空，无法训练。")

    distance = torch.tensor(
        [to_float(row.get("distance")) for row in rows],
        dtype=torch.float32,
    )
    degree_aship = torch.tensor(
        [to_float(row.get("degreeAship")) for row in rows],
        dtype=torch.float32,
    )
    obstacle_min_range = torch.tensor(
        [to_float(row.get("obstacle_min_range")) for row in rows],
        dtype=torch.float32,
    )
    obstacle_angle = torch.tensor(
        [to_float(row.get("obstacle_angle")) for row in rows],
        dtype=torch.float32,
    )

    distance_reward_raw = torch.tensor(
        [to_float(row.get("distance_reward_raw")) for row in rows],
        dtype=torch.float32,
    )
    heading_reward_raw = torch.tensor(
        [to_float(row.get("heading_reward_raw")) for row in rows],
        dtype=torch.float32,
    )
    obstacle_reward_raw = torch.tensor(
        [to_float(row.get("obstacle_reward_raw")) for row in rows],
        dtype=torch.float32,
    )
    task_reward = torch.tensor(
        [get_task_reward_from_raw(row) for row in rows],
        dtype=torch.float32,
    )
    done = torch.tensor(
        [to_float(row.get("done")) for row in rows],
        dtype=torch.float32,
    )

    # ------------------------------------------------------
    # 输入特征 features
    # ------------------------------------------------------
    if normalizer is None:
        normalizer = RewardInputNormalizer.from_raw_tensors(distance, obstacle_min_range)

    # d_goal: 到目标距离，使用固定范围归一化到 [0, 1]
    d_goal = normalizer.normalize_distance(distance)

    # theta_goal: 当前朝向/目标角度误差，归一化到 [-1, 1]
    theta_goal = angle_degree_to_minus1_1(degree_aship)

    # d_obs_min: 最近障碍物距离，使用固定范围归一化到 [0, 1]
    d_obs_min = normalizer.normalize_obstacle_min_range(obstacle_min_range)

    # theta_obs_min: 最近障碍物角度，归一化到 [-1, 1]
    theta_obs_min = angle_degree_to_minus1_1(obstacle_angle)

    features = torch.stack(
        [d_goal, theta_goal, d_obs_min, theta_obs_min],
        dim=1,
    )

    # ------------------------------------------------------
    # 四个基础奖励分量 reward_components
    # ------------------------------------------------------
    # 目标进度奖励：CSV 已经给出 distance_reward_raw（来自每步距离变化/进度）
    r_goal_progress = distance_reward_raw

    # 目标接近度奖励：越接近目标越大，使用归一化距离构造到 [0, 1]
    r_goal_proximity = 1.0 - d_goal

    # 障碍物距离奖励：CSV 已经给出 obstacle_reward_raw
    # 注意你的 obstacle_reward_raw 大多是 <= 0，表示靠近障碍物惩罚。
    r_obs_dist = obstacle_reward_raw

    # APF/航向奖励：只使用 heading_reward_raw，避免与 r_obs_dist 重复计入障碍物惩罚。
    # 如果以后你有真正的 attractive_reward / repulsive_reward，可以替换这里。
    r_apf = heading_reward_raw

    reward_components = torch.stack(
        [r_goal_progress, r_goal_proximity, r_obs_dist, r_apf],
        dim=1,
    )

    task_rewards = task_reward
    dones = done

    return features, reward_components, task_rewards, dones, normalizer


def print_dataset_info(
    features: torch.Tensor,
    reward_components: torch.Tensor,
    task_rewards: torch.Tensor,
    dones: torch.Tensor,
) -> None:
    """打印数据集基本信息，方便检查。"""
    print("========== Dataset Info ==========")
    print(f"样本数量 N: {features.shape[0]}")
    print(f"features shape: {tuple(features.shape)}")
    print(f"reward_components shape: {tuple(reward_components.shape)}")
    print(f"task_rewards shape: {tuple(task_rewards.shape)}")
    print(f"dones shape: {tuple(dones.shape)}")
    print()

    print("features 前 5 行 [d_goal, theta_goal, d_obs_min, theta_obs_min]:")
    print(features[:5])
    print()

    print("reward_components 前 5 行 [r_goal_progress, r_goal_proximity, r_obs_dist, r_apf]:")
    print(reward_components[:5])
    print()

    print("task_rewards 前 5 个:")
    print(task_rewards[:5])
    print()

    print("done 数量:", int((dones > 0.5).sum().item()))
    print("==================================")
    print()


# ==========================================================
# 3. 主训练流程
# ==========================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 读取 CSV
    features, reward_components, task_rewards, dones, normalizer = load_csv_dataset(CSV_PATH)
    normalizer.save(NORMALIZER_PATH)
    print(f"归一化参数已保存到: {NORMALIZER_PATH}")
    print_dataset_info(features, reward_components, task_rewards, dones)

    # 配置奖励权重网络
    # safe_distance=1.0 是因为 d_obs_min 已经被归一化到 [0, 1]
    config = RewardWeightConfig(
        input_dim=5,
        raw_input_dim=4,
        safe_distance=1.0,
        goal_dist_min=0.1,
        goal_dist_max=2.0,
        goal_proximity_min=0.05,
        goal_proximity_max=1.5,
        obs_dist_min=0.1,
        obs_dist_max=3.0,
        apf_min=1.0,
        apf_max=5.0,
        use_layer_norm=True,
    )

    reward_net = RewardWeightNet(config)

    # ------------------------------------------------------
    # 第一步：规则伪标签预训练
    # ------------------------------------------------------
    # 目的：让网络一开始就学到一个合理趋势：
    #   离目标远 -> 目标距离奖励权重大
    #   障碍物近 -> 障碍物/APF 权重大
    print("\n开始规则伪标签预训练...")
    pretrain_reward_net(
        reward_net=reward_net,
        feature_dataset=features,
        epochs=100,
        batch_size=32,
        lr=1e-3,
        device=device,
    )

    # ------------------------------------------------------
    # 第二步：用任务级离散奖励做相关性训练
    # ------------------------------------------------------
    # 训练目标：让
    #   w_goal_progress*r_goal_progress + w_goal_proximity*r_goal_proximity + w_obs*r_obs + w_apf*r_apf
    # 和未来任务回报 G_t 尽量正相关。
    # 注意：G_t 来自 task_reward，不再默认来自手工加权后的 total_reward。
    print("\n开始 ACWI 风格相关性训练...")
    reward_net = train_reward_net_by_correlation(
        reward_net=reward_net,
        features=features,
        reward_components=reward_components,
        task_rewards=task_rewards,
        dones=dones,
        gamma=0.99,
        epochs=200,
        batch_size=32,
        lr=5e-4,
        lambda_reg=1e-3,
        weight_ref=torch.tensor([1.0, 0.5, 1.5, 3.0], dtype=torch.float32),
        device=device,
    )

    # 保存模型
    save_reward_net(reward_net, SAVE_PATH)
    print(f"\n训练完成，模型已保存到: {SAVE_PATH}")

    # ------------------------------------------------------
    # 简单测试：输出前 10 个状态的动态权重
    # ------------------------------------------------------
    reward_net.eval()
    with torch.no_grad():
        sample_features = features[:10].to(device)
        sample_weights = reward_net(sample_features).cpu()

    print("\n前 10 个样本对应的动态奖励权重:")
    print("列含义: [w_goal_progress, w_goal_proximity, w_obs_dist, w_apf]")
    print(sample_weights)

    print("\n单个状态调用示例:")
    one_state = features[0]
    w_goal_progress, w_goal_proximity, w_obs, w_apf = reward_net.get_weights(one_state)
    print("输入状态 features[0] =", one_state.tolist())
    print("输出权重:")
    print("w_goal_progress  =", w_goal_progress)
    print("w_goal_proximity =", w_goal_proximity)
    print("w_obs_dist       =", w_obs)
    print("w_apf            =", w_apf)

    print("\n实时单步数据转换示例:")
    example_raw_step = {
        "distance": 12.0,
        "degreeAship": 30.0,
        "obstacle_min_range": 4.0,
        "obstacle_angle": -20.0,
        "distance_reward_raw": 0.1,
        "heading_reward_raw": 0.05,
        "obstacle_reward_raw": -0.2,
        "task_reward": -0.01,
        "done": 0.0,
    }
    online_features, online_reward_components, online_task_reward, online_done = raw_step_to_network_input(
        example_raw_step,
        normalizer,
    )
    print("online_features =", online_features.tolist())
    print("online_reward_components =", online_reward_components.tolist())
    print("online_task_reward =", online_task_reward)
    print("online_done =", online_done)


if __name__ == "__main__":
    main()
