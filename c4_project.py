"""
Reward Weight Network for USV Reinforcement Learning

功能：
1. 根据无人艇当前状态和障碍物风险，动态输出三个奖励参数：
   - w_goal_dist: 目标点距离奖励权重
   - w_obs_dist: 障碍物距离负奖励权重
   - w_apf: 引力斥力奖励权重

2. 将基础奖励组合为总奖励：
   R_t = w_goal_dist * r_goal_dist + w_obs_dist * r_obs_dist + w_apf * r_apf

推荐输入特征：
    x_t = [d_goal, theta_goal, d_obs_min, theta_obs_min]

网络内部会根据 d_obs_min 自动计算 risk，并拼接成：
    [d_goal, theta_goal, d_obs_min, theta_obs_min, risk]

3. 支持参考 ACWI 的相关性损失训练：
   让“加权后的奖励”与“未来任务表现/未来任务回报”正相关。

   L = -Corr(w_goal_dist*r_goal_dist + w_obs_dist*r_obs_dist + w_apf*r_apf, G_t)
       + lambda_reg * L_reg

其中：
    d_goal       : 当前无人艇到目标点距离
    theta_goal   : 目标相对无人艇航向角，建议归一化到 [-pi, pi] 或 [-1, 1]
    d_obs_min    : 最近障碍物距离
    theta_obs_min: 最近障碍物相对角度
    risk         : 碰撞风险，由网络内部根据 d_obs_min 自动计算，范围 [0, 1]
"""

import math
from dataclasses import dataclass
from collections import deque
from typing import Deque, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RewardWeightConfig:
    """奖励权重网络配置。"""

    input_dim: int = 5
    raw_input_dim: int = 4
    hidden_dim1: int = 64
    hidden_dim2: int = 64
    hidden_dim3: int = 32

    # risk = clamp(1 - d_obs_min / safe_distance, 0, 1)
    # 如果 d_obs_min 已经归一化到 [0, 1]，保持 1.0 即可
    safe_distance: float = 1.0

    # 目标点距离奖励权重范围
    goal_dist_min: float = 0.1
    goal_dist_max: float = 2.0

    # 障碍物距离负奖励权重范围
    obs_dist_min: float = 0.1
    obs_dist_max: float = 3.0

    # 引力斥力奖励权重范围
    # 注意：APF 权重下限不要太小，避免智能体忽视障碍物斥力
    apf_min: float = 0.5
    apf_max: float = 5.0

    # 是否使用 LayerNorm，提高训练稳定性
    use_layer_norm: bool = True


class RewardWeightNet(nn.Module):
    """
    动态奖励权重网络。

    输入：
        features: Tensor, shape = [batch_size, 4] 或 [batch_size, 5]
        推荐顺序：
        [d_goal, theta_goal, d_obs_min, theta_obs_min]

        如果传入 4 维，网络内部自动根据 d_obs_min 计算 risk；
        如果传入 5 维，会直接使用已有 risk。

    输出：
        weights: Tensor, shape = [batch_size, 3]
        顺序：
        [w_goal_dist, w_obs_dist, w_apf]
    """

    def __init__(self, config: Optional[RewardWeightConfig] = None):
        super().__init__()
        self.config = config or RewardWeightConfig()

        self.fc1 = nn.Linear(self.config.input_dim, self.config.hidden_dim1)
        self.fc2 = nn.Linear(self.config.hidden_dim1, self.config.hidden_dim2)
        self.fc3 = nn.Linear(self.config.hidden_dim2, self.config.hidden_dim3)
        self.out = nn.Linear(self.config.hidden_dim3, 3)

        if self.config.use_layer_norm:
            self.ln1 = nn.LayerNorm(self.config.hidden_dim1)
            self.ln2 = nn.LayerNorm(self.config.hidden_dim2)
            self.ln3 = nn.LayerNorm(self.config.hidden_dim3)
        else:
            self.ln1 = nn.Identity()
            self.ln2 = nn.Identity()
            self.ln3 = nn.Identity()

        self._init_weights()

    def _init_weights(self):
        """使用 Xavier 初始化，保证训练初期更稳定。"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)

    @staticmethod
    def _map_to_range(x: torch.Tensor, low: float, high: float) -> torch.Tensor:
        """
        将任意实数映射到 [low, high]。
        使用 sigmoid 而不是 tanh，数值更稳定。
        """
        return low + (high - low) * torch.sigmoid(x)

    def compute_risk_from_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        根据输入特征自动计算碰撞风险 risk。

        支持两种输入：
        1. [d_goal, theta_goal, d_obs_min, theta_obs_min]
           自动计算 risk 并拼接为 5 维。
        2. [d_goal, theta_goal, d_obs_min, theta_obs_min, risk]
           直接返回原输入，兼容旧数据。
        """
        if features.dim() == 1:
            features = features.unsqueeze(0)

        if features.shape[-1] == self.config.input_dim:
            return features

        if features.shape[-1] != self.config.raw_input_dim:
            raise ValueError(
                "features 最后一维必须是 4 或 5。"
                "4维为 [d_goal, theta_goal, d_obs_min, theta_obs_min]，"
                "5维为 [d_goal, theta_goal, d_obs_min, theta_obs_min, risk]。"
            )

        d_obs_min = features[:, 2:3]
        safe_distance = max(float(self.config.safe_distance), 1e-6)
        risk = torch.clamp(1.0 - d_obs_min / safe_distance, 0.0, 1.0)
        return torch.cat([features, risk], dim=-1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数：
            features: [batch_size, 4] 或 [batch_size, 5]

        返回：
            weights: [batch_size, 3]
        """
        features = self.compute_risk_from_features(features.float())

        x = F.relu(self.ln1(self.fc1(features)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))

        raw = self.out(x)

        w_goal_dist = self._map_to_range(
            raw[:, 0:1],
            self.config.goal_dist_min,
            self.config.goal_dist_max,
        )
        w_obs_dist = self._map_to_range(
            raw[:, 1:2],
            self.config.obs_dist_min,
            self.config.obs_dist_max,
        )
        w_apf = self._map_to_range(
            raw[:, 2:3],
            self.config.apf_min,
            self.config.apf_max,
        )

        weights = torch.cat([w_goal_dist, w_obs_dist, w_apf], dim=-1)
        return weights

    @torch.no_grad()
    def get_weights(self, features) -> Tuple[float, float, float]:
        """
        对单个状态输出奖励参数。

        参数：
            features: list / tuple / Tensor
                推荐传入：
                [d_goal, theta_goal, d_obs_min, theta_obs_min]

                也兼容旧格式：
                [d_goal, theta_goal, d_obs_min, theta_obs_min, risk]

        返回：
            (w_goal_dist, w_obs_dist, w_apf)
        """
        self.eval()
        if not torch.is_tensor(features):
            features = torch.tensor(features, dtype=torch.float32)
        if features.dim() == 1:
            features = features.unsqueeze(0)

        device = next(self.parameters()).device
        weights = self.forward(features.to(device)).squeeze(0).cpu().tolist()
        return weights[0], weights[1], weights[2]

    @torch.no_grad()
    def get_batch_weights(self, features: torch.Tensor) -> torch.Tensor:
        """
        对一批状态输出奖励参数。

        参数：
            features: Tensor, shape = [batch_size, 4] 或 [batch_size, 5]

        返回：
            weights: Tensor, shape = [batch_size, 3]
        """
        self.eval()
        device = next(self.parameters()).device
        return self.forward(features.to(device).float()).cpu()


class DynamicRewardScheduler:
    """
    奖励调度器。

    功能：
    1. 用 RewardWeightNet 根据状态特征输出权重；
    2. 计算目标距离奖励、障碍物距离负奖励、引力斥力奖励；
    3. 组合得到最终奖励。
    """

    def __init__(self, reward_net: RewardWeightNet, device: str = "cpu"):
        self.reward_net = reward_net.to(device)
        self.device = device

    @staticmethod
    def compute_goal_distance_reward(prev_d_goal: torch.Tensor, curr_d_goal: torch.Tensor) -> torch.Tensor:
        """
        目标点距离奖励：
        r_goal_dist = d_goal_{t-1} - d_goal_t

        如果当前更接近目标，则奖励为正；
        如果远离目标，则奖励为负。
        """
        return prev_d_goal - curr_d_goal

    @staticmethod
    def compute_obstacle_distance_reward(
        prev_d_obs_min: torch.Tensor,
        curr_d_obs_min: torch.Tensor,
    ) -> torch.Tensor:
        """
        障碍物距离负奖励：
        r_obs_dist = d_obs_t - d_obs_{t-1}

        如果当前远离障碍物，则奖励为正；
        如果当前靠近障碍物，则奖励为负。
        """
        return curr_d_obs_min - prev_d_obs_min

    @staticmethod
    def compute_apf_reward(
        attractive_reward: torch.Tensor,
        repulsive_penalty: torch.Tensor,
    ) -> torch.Tensor:
        """
        引力斥力奖励：
        r_apf = r_attractive + r_repulsive

        建议：目标点引力项为正，障碍物斥力项为负。
        """
        return attractive_reward + repulsive_penalty

    def compute_weights(self, features: torch.Tensor) -> torch.Tensor:
        """
        计算动态奖励权重。

        features shape: [batch_size, 4] 或 [batch_size, 5]
        """
        features = features.to(self.device).float()
        return self.reward_net(features)

    def compute_total_reward(
        self,
        features: torch.Tensor,
        prev_d_goal: torch.Tensor,
        curr_d_goal: torch.Tensor,
        prev_d_obs_min: torch.Tensor,
        curr_d_obs_min: torch.Tensor,
        attractive_reward: torch.Tensor,
        repulsive_penalty: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算总奖励。

        返回：
            total_reward: [batch_size, 1]
            info: 包含各项奖励和权重，方便调试与可视化
        """
        features = features.to(self.device).float()
        prev_d_goal = prev_d_goal.to(self.device).float().view(-1, 1)
        curr_d_goal = curr_d_goal.to(self.device).float().view(-1, 1)
        prev_d_obs_min = prev_d_obs_min.to(self.device).float().view(-1, 1)
        curr_d_obs_min = curr_d_obs_min.to(self.device).float().view(-1, 1)
        attractive_reward = attractive_reward.to(self.device).float().view(-1, 1)
        repulsive_penalty = repulsive_penalty.to(self.device).float().view(-1, 1)

        weights = self.compute_weights(features)

        w_goal_dist = weights[:, 0:1]
        w_obs_dist = weights[:, 1:2]
        w_apf = weights[:, 2:3]

        r_goal_dist = self.compute_goal_distance_reward(prev_d_goal, curr_d_goal)
        r_obs_dist = self.compute_obstacle_distance_reward(prev_d_obs_min, curr_d_obs_min)
        r_apf = self.compute_apf_reward(attractive_reward, repulsive_penalty)

        total_reward = (
            w_goal_dist * r_goal_dist
            + w_obs_dist * r_obs_dist
            + w_apf * r_apf
        )

        info = {
            "w_goal_dist": w_goal_dist.detach(),
            "w_obs_dist": w_obs_dist.detach(),
            "w_apf": w_apf.detach(),
            "r_goal_dist": r_goal_dist.detach(),
            "r_obs_dist": r_obs_dist.detach(),
            "r_apf": r_apf.detach(),
            "total_reward": total_reward.detach(),
        }

        return total_reward, info


# ==========================================================
# 可选：用规则生成伪标签，预训练 RewardWeightNet
# ==========================================================

def generate_rule_based_weight_labels(features: torch.Tensor) -> torch.Tensor:
    """
    根据人工规则生成伪标签，用于预训练奖励权重网络。

    输入 features 顺序：
        推荐：[d_goal, theta_goal, d_obs_min, theta_obs_min]
        兼容：[d_goal, theta_goal, d_obs_min, theta_obs_min, risk]

    输出 labels：
        [w_goal_dist_label, w_obs_dist_label, w_apf_label]

    说明：
    - 距离目标较远时，提高目标点距离奖励权重；
    - 障碍物较近时，提高障碍物距离负奖励权重；
    - 碰撞风险较高时，提高引力斥力奖励权重。
    """
    if features.dim() == 1:
        features = features.unsqueeze(0)

    if features.shape[-1] == 4:
        d_obs_min_for_risk = features[:, 2:3]
        risk = torch.clamp(1.0 - d_obs_min_for_risk, 0.0, 1.0)
        features = torch.cat([features, risk], dim=-1)
    elif features.shape[-1] != 5:
        raise ValueError("features 最后一维必须是 4 或 5。")

    d_goal = features[:, 0:1]
    d_obs_min = features[:, 2:3]
    risk = features[:, 4:5]

    # 这里假设 d_goal 和 d_obs_min 已经过合理归一化
    # d_goal 越大，越需要目标点距离奖励引导
    w_goal_dist = 0.1 + 1.9 * torch.sigmoid(2.0 * (d_goal - 0.5))

    # 障碍物越近，越需要提高障碍物距离负奖励权重
    obs_risk = torch.clamp(1.0 - d_obs_min, 0.0, 1.0)
    w_obs_dist = 0.1 + 2.9 * obs_risk

    # 风险越高，越需要强调 APF 中的斥力避障项
    apf_risk = torch.clamp(0.5 * obs_risk + 0.5 * risk, 0.0, 1.0)
    w_apf = 0.5 + 4.5 * apf_risk

    labels = torch.cat([w_goal_dist, w_obs_dist, w_apf], dim=-1)
    return labels


def pretrain_reward_net(
    reward_net: RewardWeightNet,
    feature_dataset: torch.Tensor,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu",
):
    """
    使用规则伪标签预训练奖励权重网络。

    feature_dataset shape: [num_samples, 4] 或 [num_samples, 5]
    """
    reward_net.to(device)
    reward_net.train()

    optimizer = torch.optim.Adam(reward_net.parameters(), lr=lr)
    num_samples = feature_dataset.shape[0]

    for epoch in range(epochs):
        perm = torch.randperm(num_samples)
        epoch_loss = 0.0

        for start in range(0, num_samples, batch_size):
            idx = perm[start:start + batch_size]
            batch_features = feature_dataset[idx].to(device).float()
            target_weights = generate_rule_based_weight_labels(batch_features).to(device)

            pred_weights = reward_net(batch_features)
            loss = F.mse_loss(pred_weights, target_weights)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reward_net.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += loss.item() * batch_features.shape[0]

        epoch_loss /= num_samples

        if (epoch + 1) % 20 == 0:
            print(f"[Pretrain] Epoch {epoch + 1:03d}/{epochs}, Loss = {epoch_loss:.6f}")


# ==========================================================
# 参考 ACWI：用未来任务表现训练奖励参数网络
# ==========================================================

def compute_weighted_reward(
    weights: torch.Tensor,
    reward_components: torch.Tensor,
) -> torch.Tensor:
    """
    根据网络输出的奖励参数，计算加权奖励。

    参数：
        weights: Tensor, shape = [batch_size, 3]
            [w_goal_dist, w_obs_dist, w_apf]

        reward_components: Tensor, shape = [batch_size, 3]
            [r_goal_dist, r_obs_dist, r_apf]

    返回：
        weighted_reward: Tensor, shape = [batch_size, 1]
            w_goal_dist*r_goal_dist + w_obs_dist*r_obs_dist + w_apf*r_apf
    """
    return torch.sum(weights * reward_components, dim=-1, keepdim=True)


def compute_discounted_returns(
    rewards: torch.Tensor,
    dones: Optional[torch.Tensor] = None,
    gamma: float = 0.99,
) -> torch.Tensor:
    """
    计算未来折现任务回报 G_t。

    参数：
        rewards: Tensor, shape = [T] 或 [T, 1]
            任务奖励序列。建议是独立任务评价，例如：
            到达目标 +100，碰撞 -100，超时 -10，每步 -0.01。

        dones: Tensor, shape = [T] 或 [T, 1]
            是否结束。为 None 时，默认整段数据属于同一条未截断轨迹。

        gamma: float
            折扣因子。

    返回：
        returns: Tensor, shape = [T, 1]
    """
    rewards = rewards.float().view(-1)
    if dones is None:
        dones = torch.zeros_like(rewards)
    else:
        dones = dones.float().view(-1)

    returns = torch.zeros_like(rewards)
    running_return = torch.tensor(0.0, device=rewards.device, dtype=rewards.dtype)

    for t in reversed(range(rewards.shape[0])):
        if dones[t] > 0.5:
            running_return = torch.tensor(0.0, device=rewards.device, dtype=rewards.dtype)
        running_return = rewards[t] + gamma * running_return
        returns[t] = running_return

    return returns.view(-1, 1)


def reward_weight_correlation_loss(
    weights: torch.Tensor,
    reward_components: torch.Tensor,
    future_returns: torch.Tensor,
    weight_ref: Optional[torch.Tensor] = None,
    lambda_reg: float = 1e-3,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    参考 ACWI 的奖励参数网络损失函数。

    核心思想：
        让“当前权重加权后的奖励”与“未来任务回报”正相关。

    数学形式：
        weighted_reward_t = w_goal_dist*r_goal_dist + w_obs_dist*r_obs_dist + w_apf*r_apf

        L_corr = -Corr(weighted_reward_t, G_t)

        L_reg = mean((log(w) - log(w_ref))^2)

        L = L_corr + lambda_reg * L_reg

    参数：
        weights: Tensor, shape = [batch_size, 3]
            奖励参数网络输出：[w_goal_dist, w_obs_dist, w_apf]

        reward_components: Tensor, shape = [batch_size, 3]
            基础奖励分量：[r_goal_dist, r_obs_dist, r_apf]

        future_returns: Tensor, shape = [batch_size] 或 [batch_size, 1]
            未来任务回报 G_t。

        weight_ref: Tensor, shape = [3]
            权重参考值。默认 [1.0, 1.5, 3.0]。
            APF 参考值更大，是为了强调斥力避障安全性。

        lambda_reg: float
            正则化系数。

    返回：
        loss: Tensor
            用于反向传播的损失。

        info: dict
            调试信息。
    """
    if future_returns.dim() == 1:
        future_returns = future_returns.view(-1, 1)

    reward_components = reward_components.float()
    future_returns = future_returns.float()

    weighted_reward = compute_weighted_reward(weights, reward_components)

    weighted_reward_norm = (
        weighted_reward - weighted_reward.mean()
    ) / (weighted_reward.std(unbiased=False) + eps)

    future_returns_norm = (
        future_returns - future_returns.mean()
    ) / (future_returns.std(unbiased=False) + eps)

    corr = torch.mean(weighted_reward_norm * future_returns_norm)
    loss_corr = -corr

    if weight_ref is None:
        weight_ref = torch.tensor([1.0, 1.5, 3.0], device=weights.device, dtype=weights.dtype)
    else:
        weight_ref = weight_ref.to(device=weights.device, dtype=weights.dtype)
    weight_ref = weight_ref.view(1, -1)

    loss_reg = torch.mean((torch.log(weights + eps) - torch.log(weight_ref + eps)) ** 2)
    loss = loss_corr + lambda_reg * loss_reg

    info = {
        "loss": loss.detach(),
        "loss_corr": loss_corr.detach(),
        "loss_reg": loss_reg.detach(),
        "corr": corr.detach(),
        "weighted_reward_mean": weighted_reward.mean().detach(),
        "weighted_reward_std": weighted_reward.std(unbiased=False).detach(),
        "future_return_mean": future_returns.mean().detach(),
        "future_return_std": future_returns.std(unbiased=False).detach(),
    }
    return loss, info


def train_reward_net_by_correlation(
    reward_net: RewardWeightNet,
    features: torch.Tensor,
    reward_components: torch.Tensor,
    task_rewards: Optional[torch.Tensor] = None,
    future_returns: Optional[torch.Tensor] = None,
    dones: Optional[torch.Tensor] = None,
    gamma: float = 0.99,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 5e-4,
    lambda_reg: float = 1e-3,
    weight_ref: Optional[torch.Tensor] = None,
    device: str = "cpu",
) -> RewardWeightNet:
    """
    使用 ACWI 风格的相关性损失训练奖励参数网络。

    你需要提供以下数据：
        features: [N, 4] 或 [N, 5]
            输入经验特征，例如：
            推荐：[d_goal, theta_goal, d_obs_min, theta_obs_min]
            兼容：[d_goal, theta_goal, d_obs_min, theta_obs_min, risk]

        reward_components: [N, 3]
            三个基础奖励分量：
            [r_goal_dist, r_obs_dist, r_apf]

        task_rewards 或 future_returns 二选一：
            task_rewards: [N]
                原始任务表现奖励，用它计算 future_returns。
            future_returns: [N]
                如果你已经提前算好了未来任务回报，可以直接传入。

    训练目标：
        最小化：
        L = -Corr(
              w_goal_dist*r_goal_dist + w_obs_dist*r_obs_dist + w_apf*r_apf,
              G_t
            ) + lambda_reg * L_reg

    返回：
        训练后的 reward_net。
    """
    reward_net.to(device)
    reward_net.train()

    features = features.to(device).float()
    reward_components = reward_components.to(device).float()

    if future_returns is None:
        if task_rewards is None:
            raise ValueError("task_rewards 和 future_returns 至少需要提供一个。")
        task_rewards = task_rewards.to(device).float()
        dones = None if dones is None else dones.to(device).float()
        future_returns = compute_discounted_returns(task_rewards, dones=dones, gamma=gamma)
    else:
        future_returns = future_returns.to(device).float().view(-1, 1)

    if not (features.shape[0] == reward_components.shape[0] == future_returns.shape[0]):
        raise ValueError("features、reward_components、future_returns 的样本数量必须一致。")

    optimizer = torch.optim.Adam(reward_net.parameters(), lr=lr, weight_decay=1e-6)
    num_samples = features.shape[0]

    for epoch in range(epochs):
        perm = torch.randperm(num_samples, device=device)
        epoch_loss = 0.0
        epoch_corr = 0.0

        for start in range(0, num_samples, batch_size):
            idx = perm[start:start + batch_size]
            batch_features = features[idx]
            batch_reward_components = reward_components[idx]
            batch_future_returns = future_returns[idx]

            weights = reward_net(batch_features)

            loss, info = reward_weight_correlation_loss(
                weights=weights,
                reward_components=batch_reward_components,
                future_returns=batch_future_returns,
                weight_ref=weight_ref,
                lambda_reg=lambda_reg,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reward_net.parameters(), max_norm=1.0)
            optimizer.step()

            batch_size_actual = batch_features.shape[0]
            epoch_loss += loss.item() * batch_size_actual
            epoch_corr += info["corr"].item() * batch_size_actual

        epoch_loss /= num_samples
        epoch_corr /= num_samples

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(
                f"[CorrTrain] Epoch {epoch + 1:03d}/{epochs}, "
                f"Loss = {epoch_loss:.6f}, Corr = {epoch_corr:.6f}"
            )

    return reward_net


class OnlineRewardWeightUpdater:
    """
    在线奖励权重网络更新器。

    主网络每交互一步，可以调用 add_experience() 写入新经验；
    每隔 update_interval 步，调用 maybe_update() 自动判断是否微调 RewardWeightNet。

    缓存中的单条经验包含：
        features: [d_goal, theta_goal, d_obs_min, theta_obs_min]
        reward_components: [r_goal_dist, r_obs_dist, r_apf]
        task_reward: 主任务奖励
        done: episode 是否结束
    """

    def __init__(
        self,
        reward_net: RewardWeightNet,
        device: str = "cpu",
        buffer_size: int = 5000,
        min_buffer_size: int = 128,
        update_interval: int = 50,
        batch_size: int = 64,
        update_epochs: int = 3,
        lr: float = 1e-4,
        gamma: float = 0.99,
        lambda_reg: float = 1e-3,
        weight_ref: Optional[torch.Tensor] = None,
        grad_clip_norm: float = 1.0,
    ):
        self.reward_net = reward_net.to(device)
        self.device = device
        self.buffer_size = buffer_size
        self.min_buffer_size = min_buffer_size
        self.update_interval = update_interval
        self.batch_size = batch_size
        self.update_epochs = update_epochs
        self.lr = lr
        self.gamma = gamma
        self.lambda_reg = lambda_reg
        self.weight_ref = weight_ref
        self.grad_clip_norm = grad_clip_norm

        self.buffer: Deque[Tuple[torch.Tensor, torch.Tensor, float, float]] = deque(maxlen=buffer_size)
        self.total_steps = 0
        self.update_times = 0
        self.optimizer = torch.optim.Adam(self.reward_net.parameters(), lr=lr, weight_decay=1e-6)

    @torch.no_grad()
    def get_weights(self, features) -> Tuple[float, float, float]:
        """在线推理接口：输入当前状态，输出当前奖励权重。"""
        return self.reward_net.get_weights(features)

    def add_experience(
        self,
        features,
        reward_components,
        task_reward: float,
        done: float = 0.0,
    ) -> None:
        """向在线缓存加入一条新经验。"""
        if not torch.is_tensor(features):
            features = torch.tensor(features, dtype=torch.float32)
        if not torch.is_tensor(reward_components):
            reward_components = torch.tensor(reward_components, dtype=torch.float32)

        self.buffer.append((
            features.detach().cpu().float().view(-1),
            reward_components.detach().cpu().float().view(-1),
            float(task_reward),
            float(done),
        ))
        self.total_steps += 1

    def ready_to_update(self) -> bool:
        """判断当前是否满足在线更新条件。"""
        if len(self.buffer) < self.min_buffer_size:
            return False
        return self.total_steps % self.update_interval == 0

    def maybe_update(self) -> Optional[Dict[str, float]]:
        """达到更新频率时执行微调，否则返回 None。"""
        if not self.ready_to_update():
            return None
        return self.update()

    def _buffer_to_tensors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features, reward_components, task_rewards, dones = zip(*self.buffer)
        return (
            torch.stack(list(features)).to(self.device).float(),
            torch.stack(list(reward_components)).to(self.device).float(),
            torch.tensor(task_rewards, device=self.device, dtype=torch.float32),
            torch.tensor(dones, device=self.device, dtype=torch.float32),
        )

    def update(self) -> Dict[str, float]:
        """用缓存中的最近经验对奖励权重网络做小步在线微调。"""
        self.reward_net.train()
        features, reward_components, task_rewards, dones = self._buffer_to_tensors()
        future_returns = compute_discounted_returns(task_rewards, dones=dones, gamma=self.gamma)
        num_samples = features.shape[0]

        last_loss = 0.0
        last_corr = 0.0

        for _ in range(self.update_epochs):
            sample_size = min(self.batch_size, num_samples)
            idx = torch.randint(0, num_samples, (sample_size,), device=self.device)

            batch_features = features[idx]
            batch_reward_components = reward_components[idx]
            batch_future_returns = future_returns[idx]

            weights = self.reward_net(batch_features)
            loss, info = reward_weight_correlation_loss(
                weights=weights,
                reward_components=batch_reward_components,
                future_returns=batch_future_returns,
                weight_ref=self.weight_ref,
                lambda_reg=self.lambda_reg,
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.reward_net.parameters(), max_norm=self.grad_clip_norm)
            self.optimizer.step()

            last_loss = float(loss.detach().cpu().item())
            last_corr = float(info["corr"].detach().cpu().item())

        self.update_times += 1
        self.reward_net.eval()

        return {
            "loss": last_loss,
            "corr": last_corr,
            "buffer_size": float(len(self.buffer)),
            "total_steps": float(self.total_steps),
            "update_times": float(self.update_times),
        }


def save_reward_net(reward_net: RewardWeightNet, path: str) -> None:
    """保存训练后的奖励参数网络。"""
    torch.save(
        {
            "config": reward_net.config,
            "model_state_dict": reward_net.state_dict(),
        },
        path,
    )


def load_reward_net(path: str, device: str = "cpu") -> RewardWeightNet:
    """加载训练后的奖励参数网络。"""
    checkpoint = torch.load(path, map_location=device)
    reward_net = RewardWeightNet(checkpoint["config"])
    reward_net.load_state_dict(checkpoint["model_state_dict"])
    reward_net.to(device)
    reward_net.eval()
    return reward_net


# ==========================================================
# 示例：如何在训练循环中使用
# ==========================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = RewardWeightConfig(
        input_dim=5,
        raw_input_dim=4,
        safe_distance=1.0,
        goal_dist_min=0.1,
        goal_dist_max=2.0,
        obs_dist_min=0.1,
        obs_dist_max=3.0,
        apf_min=0.5,
        apf_max=5.0,
    )

    reward_net = RewardWeightNet(config)
    scheduler = DynamicRewardScheduler(reward_net, device=device)

    # 假设 batch_size = 4
    # features = [d_goal, theta_goal, d_obs_min, theta_obs_min]
    # risk 不需要主代码提供，网络会由 d_obs_min 自动计算
    features = torch.tensor([
        [0.8, 0.20, 0.9,  0.10],   # 离目标较远，障碍风险低
        [0.3, 1.00, 0.8, -0.20],   # 角度偏差较大
        [0.5, 0.30, 0.2,  0.05],   # 障碍物很近，风险高
        [0.1, 0.10, 0.7,  0.40],   # 接近目标，风险低
    ], dtype=torch.float32)

    prev_d_goal = torch.tensor([0.85, 0.35, 0.55, 0.12])
    curr_d_goal = torch.tensor([0.80, 0.30, 0.58, 0.08])
    prev_d_obs_min = torch.tensor([0.85, 0.75, 0.25, 0.75])
    curr_d_obs_min = features[:, 2]
    attractive_reward = prev_d_goal - curr_d_goal
    repulsive_penalty = -torch.clamp(1.0 - curr_d_obs_min, min=0.0) ** 2

    total_reward, info = scheduler.compute_total_reward(
        features=features,
        prev_d_goal=prev_d_goal,
        curr_d_goal=curr_d_goal,
        prev_d_obs_min=prev_d_obs_min,
        curr_d_obs_min=curr_d_obs_min,
        attractive_reward=attractive_reward,
        repulsive_penalty=repulsive_penalty,
    )

    print("动态权重：")
    print(torch.cat([info["w_goal_dist"], info["w_obs_dist"], info["w_apf"]], dim=-1))

    print("基础奖励：")
    print(torch.cat([info["r_goal_dist"], info["r_obs_dist"], info["r_apf"]], dim=-1))

    print("总奖励：")
    print(total_reward)

    # ======================================================
    # ACWI 风格相关性训练示例
    # 实际使用时，把下面的示例数据换成你们采集到的一批经验。
    # ======================================================

    # reward_components = [r_goal_dist, r_obs_dist, r_apf]
    reward_components = torch.cat(
        [info["r_goal_dist"], info["r_obs_dist"], info["r_apf"]],
        dim=-1,
    )

    # task_rewards 是独立任务表现，不建议直接等于加权奖励。
    # 示例：到达目标 +100，碰撞 -100，普通步 -0.01。
    task_rewards = torch.tensor([-0.01, -0.01, -0.01, -100.0], dtype=torch.float32)
    dones = torch.tensor([0, 0, 0, 1], dtype=torch.float32)

    print("\n开始 ACWI 风格相关性训练示例：")
    reward_net = train_reward_net_by_correlation(
        reward_net=reward_net,
        features=features,
        reward_components=reward_components,
        task_rewards=task_rewards,
        dones=dones,
        gamma=0.99,
        epochs=5,
        batch_size=4,
        lr=5e-4,
        lambda_reg=1e-3,
        weight_ref=torch.tensor([1.0, 1.5, 3.0]),
        device=device,
    )

    print("\n训练后批量输出奖励参数：")
    print(reward_net.get_batch_weights(features))

    print("\n训练后单个状态输出奖励参数：")
    print(reward_net.get_weights([0.4, 0.7, 0.2, 0.1]))
