import logging
import torch
import torch.nn as nn
import numpy as np
import math
from torch.distributions import Categorical, MultivariateNormal

from usvlib4ros.user.tensorboard_logging import TensorBoardMetricsWriter

logger = logging.getLogger(__name__)

# Device configuration
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class RolloutBuffer:
    """经验回放缓冲区,存储PPO训练所需的转移数据。
    
    在一个轨迹片段(trajectory)内收集状态、动作、对数概率、奖励和终止标志,
    用于后续的 PPO 策略更新。
    """

    def __init__(self):
        """初始化缓冲区列表。"""
        self.actions = []      # 动作列表(标量或向量)
        self.states = []       # 状态列表(向量)
        self.logprobs = []     # 动作对数概率列表
        self.state_values = [] # 状态价值估计列表
        self.rewards = []      # 即时奖励列表
        self.is_terminals = [] # 终止标志列表(True/False,表示是否为终止状态)

    def clear(self):
        """清空缓冲区中的所有数据,为下一个训练片段做准备。"""
        self.actions.clear()
        self.states.clear()
        self.logprobs.clear()
        self.state_values.clear()
        self.rewards.clear()
        self.is_terminals.clear()


class ActorCritic(nn.Module):
    """改进版 Actor-Critic：共享特征 + LayerNorm + SiLU + 正交初始化。"""

    def __init__(self, state_dim: int, action_dim: int,
                 has_continuous_action_space: bool, action_std_init: float):
        super().__init__()
        self.has_continuous_action_space = has_continuous_action_space
        hidden_dim = 256

        # ===== 共享特征提取器 =====
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        # ===== Actor 输出头 =====
        self.actor_mean = nn.Linear(hidden_dim, action_dim)

        # ===== Critic 输出头 =====
        self.critic = nn.Linear(hidden_dim, 1)

        # ===== 连续动作对数标准差 =====
        if has_continuous_action_space:
            self.action_dim = action_dim
            self.log_std = nn.Parameter(
                torch.full((action_dim,), math.log(action_std_init), dtype=torch.float32)
            )

        self._init_weights()

    def _init_weights(self):
        # 共享层 gain=np.sqrt(2)
        for module in self.shared.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0.0)

        # Actor 末层 gain=0.01
        torch.nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        if self.actor_mean.bias is not None:
            torch.nn.init.constant_(self.actor_mean.bias, 0.0)

        # Critic 末层 gain=1.0
        torch.nn.init.orthogonal_(self.critic.weight, gain=1.0)
        if self.critic.bias is not None:
            torch.nn.init.constant_(self.critic.bias, 0.0)

    def _get_log_std(self, learnable: bool) -> torch.Tensor:
        log_std = torch.clamp(self.log_std, min=-3.0, max=-0.3)
        return log_std if learnable else log_std.detach()

    def _get_action_mean(self, raw_action: torch.Tensor) -> torch.Tensor:
        raw_action = self._sanitize_tensor(raw_action)
        return torch.tanh(raw_action)   # 统一 [-1, 1]

    @staticmethod
    def _sanitize_tensor(t: torch.Tensor) -> torch.Tensor:
        return torch.where(
            torch.isnan(t) | torch.isinf(t), torch.zeros_like(t), t
        )

    def act(self, state: torch.Tensor):
        state = self._sanitize_tensor(state.to(device))
        shared_out = self.shared(state)

        if self.has_continuous_action_space:
            raw = self.actor_mean(shared_out)
            action_mean = self._get_action_mean(raw)
            log_std = self._get_log_std(learnable=True)
            std = torch.exp(log_std)
            scale_tril = torch.diag(std)
            dist = MultivariateNormal(action_mean, scale_tril=scale_tril)
        else:
            logits = self.actor_mean(shared_out)
            dist = Categorical(logits=logits)

        action = dist.sample()
        return action.detach(), dist.log_prob(action).detach()

    def evaluate(self, state: torch.Tensor, action: torch.Tensor, learnable_std: bool = True):
        state = self._sanitize_tensor(state.to(device))
        shared_out = self.shared(state)

        if self.has_continuous_action_space:
            raw = self.actor_mean(shared_out)
            action_mean = self._get_action_mean(raw)
            action_mean = self._sanitize_tensor(action_mean)
            log_std = self._get_log_std(learnable=learnable_std)
            std = torch.exp(log_std)
            std_batch = std.unsqueeze(0).expand(action_mean.size(0), -1)
            scale_tril = torch.diag_embed(std_batch)
            dist = MultivariateNormal(action_mean, scale_tril=scale_tril)
        else:
            logits = self.actor_mean(shared_out)
            dist = Categorical(logits=logits)

        state_value = self.critic(shared_out)
        return dist.log_prob(action), state_value, dist.entropy()

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """外部调用获得状态价值，内部自动经过共享层。"""
        state = self._sanitize_tensor(state.to(device))
        shared_out = self.shared(state)
        return self.critic(shared_out)


class PPO:
    """PPO(Proximal Policy Optimization)算法实现。

    支持离散和连续动作空间,使用裁剪(clipping)的替代目标函数,
    并通过多个 epoch 对同一批数据进行更新。
    """

    def __init__(self, state_dim: int, action_dim: int,
                 lr_actor: float, lr_critic: float,
                 gamma: float, K_epochs: int, eps_clip: float,
                 has_continuous_action_space: bool, action_std_init: float,
                 writer=None, gae_lambda: float = 0.95):
        """初始化 PPO 算法的超参数、网络和优化器。

        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            lr_actor: Actor 网络的学习率
            lr_critic: Critic 网络的学习率
            gamma: 折扣因子(reward discount)
            K_epochs: 每次策略更新时,对同一批数据迭代优化的 epoch 数
            eps_clip: PPO 裁剪范围(通常 0.2)
            has_continuous_action_space: 是否连续动作空间
            action_std_init: 连续动作空间下初始标准差
        """
        self.has_continuous_action_space = has_continuous_action_space
        if has_continuous_action_space:
            self.action_std = action_std_init   # 保存初始标准差,但代码中未使用动态调整

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.buffer = RolloutBuffer()           # 经验缓冲区

        # 当前策略网络(用于训练)
        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        # 优化器:Actor 和 Critic 使用各自的学习率(通过参数分组实现)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.shared.parameters(), 'lr': lr_actor},
            {'params': self.policy.actor_mean.parameters(), 'lr': lr_actor},
            {'params': [self.policy.log_std], 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic},
        ])

        # 旧策略网络(用于采样,更新时与当前策略比较)
        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.mse_loss = nn.MSELoss()   # 用于 Critic 的损失函数
        self.tb_writer = TensorBoardMetricsWriter(writer=writer) if writer is not None else None
        self.update_step = 0
        self.min_update_buffer_size = 128

    @staticmethod
    def _sanitize_tensor(t: torch.Tensor) -> torch.Tensor:
        """清理 PPO 入口张量中的 NaN/Inf。"""
        return torch.where(
            torch.isnan(t) | torch.isinf(t), torch.zeros_like(t), t
        )

    def select_action(self, state) -> int:
        state = self._sanitize_tensor(state.to(device))
        with torch.no_grad():
            action, logprob = self.policy_old.act(state)
            # 修改这一行：
            state_value = self.policy_old.get_value(state).squeeze(-1)

        self.buffer.states.append(state.cpu().numpy().tolist())
        self.buffer.actions.append(action.cpu().numpy().tolist())
        self.buffer.logprobs.append(logprob.cpu().numpy().tolist())
        self.buffer.state_values.append(float(state_value.item()))

        return action.cpu().numpy() if self.has_continuous_action_space else action.item()

    def update(self):
        """使用缓冲区中收集的经验更新策略网络(PPO 核心更新步骤)。"""
        if not self.buffer.rewards:
            return None
        if len(self.buffer.rewards) < self.min_update_buffer_size:
            return None

        # ========== 1. 将缓冲区中的列表数据转换为张量 ==========
        # 状态:从列表转换并堆叠
        old_states = torch.tensor(self.buffer.states, dtype=torch.float32, device=device).detach()
        # 动作
        old_actions = torch.tensor(self.buffer.actions, dtype=torch.float32, device=device).detach()
        if self.has_continuous_action_space and old_actions.dim() == 1:
            old_actions = old_actions.unsqueeze(-1)
        # 旧对数概率
        old_logprobs = torch.tensor(self.buffer.logprobs, dtype=torch.float32, device=device).detach()
        old_state_values = torch.tensor(self.buffer.state_values, dtype=torch.float32, device=device).detach()

        # ========== 2. 使用真实 GAE 计算优势和 returns ==========
        rewards = torch.tensor(self.buffer.rewards, dtype=torch.float32, device=device)
        terminals = torch.tensor(self.buffer.is_terminals, dtype=torch.float32, device=device)

        advantages = torch.zeros_like(rewards)
        gae = torch.tensor(0.0, dtype=torch.float32, device=device)
        next_value = torch.tensor(0.0, dtype=torch.float32, device=device)

        for step in reversed(range(len(rewards))):
            mask = 1.0 - terminals[step]
            delta = rewards[step] + self.gamma * next_value * mask - old_state_values[step]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[step] = gae
            next_value = old_state_values[step]

        returns = advantages + old_state_values
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-7)

        # ========== 3. 多次 epoch 更新策略 ==========
        actor_loss_value = 0.0
        critic_loss_value = 0.0
        total_loss_value = 0.0
        entropy_value = 0.0
        ratio_mean_value = 0.0
        advantage_mean_value = advantages.mean().item()
        for _ in range(self.K_epochs):
            # 在当前策略下评估这批数据
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = state_values.squeeze(-1)
                
            # 计算概率比 r(θ) = π_θ(a|s) / π_old(a|s)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # PPO 裁剪损失
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2)
            critic_loss = self.mse_loss(state_values, returns)
            entropy_bonus = dist_entropy
            # 总损失 = -min(surr1, surr2) + 0.5 * (V - G)^2 - 0.01 * 熵
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_bonus

            if not torch.isfinite(loss).all():
                logger.warning("PPO update skipped because loss contains NaN/Inf")
                self.buffer.clear()
                return None

            actor_loss_value = actor_loss.mean().item()
            critic_loss_value = critic_loss.item()
            total_loss_value = loss.mean().item()
            entropy_value = entropy_bonus.mean().item()
            ratio_mean_value = ratios.mean().item()
            advantage_mean_value = advantages.mean().item()

            # 反向传播与参数更新
            self.optimizer.zero_grad()
            loss.mean().backward()
            # 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            self.optimizer.step()

        if self.tb_writer is not None:
            self.tb_writer.log_update(
                self.update_step,
                total_loss_value,
                actor_loss_value,
                critic_loss_value,
                entropy_value,
                ratio_mean_value,
                advantage_mean_value,
            )
        self.update_step += 1

        # 更新完成后,将旧策略网络同步为当前策略网络
        self.policy_old.load_state_dict(self.policy.state_dict())
        # 清空缓冲区,准备下一轮收集
        self.buffer.clear()
        return {
            "actor_loss": actor_loss_value,
            "critic_loss": critic_loss_value,
            "total_loss": total_loss_value,
            "entropy": entropy_value,
            "buffer_size": len(rewards),
        }

    def load(self, checkpoint_path: str):
        """从文件加载模型权重(同时更新 policy 和 policy_old)。"""
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=device))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=device))

    def save(self, checkpoint_path: str):
        """将当前策略网络的权重保存到文件。"""
        torch.save(self.policy.state_dict(), checkpoint_path)
