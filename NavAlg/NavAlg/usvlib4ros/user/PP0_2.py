import logging
import torch
import torch.nn as nn
import numpy as np
import math
from torch.distributions import Categorical, MultivariateNormal

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
        self.next_states = []  # 下一时刻状态列表,用于GRU辅助预测损失
        self.rewards = []      # 即时奖励列表
        self.is_terminals = [] # 终止标志列表(True/False,表示是否为终止状态)

    def clear(self):
        """清空缓冲区中的所有数据,为下一个训练片段做准备。"""
        self.actions.clear()
        self.states.clear()
        self.logprobs.clear()
        self.state_values.clear()
        self.next_states.clear()
        self.rewards.clear()
        self.is_terminals.clear()


class ActorCritic(nn.Module):
    """Recurrent Actor-Critic: actor/critic 各自使用独立 GRU。"""

    def __init__(self, state_dim: int, action_dim: int,
                 has_continuous_action_space: bool, action_std_init: float,
                 gru_input_dim: int = 128, gru_hidden_dim: int = 256,
                 aux_hidden_dim: int = 128):
        super().__init__()
        self.has_continuous_action_space = has_continuous_action_space
        self.state_dim = state_dim
        self.action_dim = action_dim

        # ===== Actor recurrent 分支 =====
        self.actor_input = nn.Linear(state_dim, gru_input_dim)
        self.actor_gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=gru_hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.actor_head = nn.Sequential(
            nn.LayerNorm(gru_hidden_dim),
            nn.Linear(gru_hidden_dim, gru_hidden_dim),
            nn.LayerNorm(gru_hidden_dim),
            nn.SiLU(),
            nn.Linear(gru_hidden_dim, action_dim),
        )

        # ===== Critic recurrent 分支 =====
        self.critic_input = nn.Linear(state_dim, gru_input_dim)
        self.critic_gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=gru_hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.critic_head = nn.Sequential(
            nn.LayerNorm(gru_hidden_dim),
            nn.Linear(gru_hidden_dim, gru_hidden_dim),
            nn.LayerNorm(gru_hidden_dim),
            nn.SiLU(),
            nn.Linear(gru_hidden_dim, 1),
        )

        # ===== 辅助预测头: 融合 actor/critic 时序特征预测 next_state =====
        self.aux_predictor = nn.Sequential(
            nn.Linear(gru_hidden_dim * 2, aux_hidden_dim),
            nn.LayerNorm(aux_hidden_dim),
            nn.SiLU(),
            nn.Linear(aux_hidden_dim, state_dim),
        )

        # ===== 连续动作对数标准差 =====
        if has_continuous_action_space:
            self.log_std = nn.Parameter(
                torch.full((action_dim,), math.log(action_std_init), dtype=torch.float32)
            )

        self._init_weights()

    def actor_parameters(self):
        for module in (self.actor_input, self.actor_gru, self.actor_head):
            yield from module.parameters()

    def critic_parameters(self):
        for module in (self.critic_input, self.critic_gru, self.critic_head):
            yield from module.parameters()

    def aux_parameters(self):
        yield from self.aux_predictor.parameters()

    def _init_weights(self):
        for module in (
            self.actor_input,
            self.actor_head,
            self.critic_input,
            self.critic_head,
            self.aux_predictor,
        ):
            for child in module.modules():
                if isinstance(child, nn.Linear):
                    torch.nn.init.orthogonal_(child.weight, gain=np.sqrt(2))
                    if child.bias is not None:
                        torch.nn.init.constant_(child.bias, 0.0)

        for gru in (self.actor_gru, self.critic_gru):
            for name, param in gru.named_parameters():
                if "weight_ih" in name:
                    torch.nn.init.xavier_uniform_(param)
                elif "weight_hh" in name:
                    torch.nn.init.orthogonal_(param)
                elif "bias" in name:
                    torch.nn.init.constant_(param, 0.0)

        torch.nn.init.orthogonal_(self.actor_head[-1].weight, gain=0.01)
        if self.actor_head[-1].bias is not None:
            torch.nn.init.constant_(self.actor_head[-1].bias, 0.0)
        torch.nn.init.orthogonal_(self.critic_head[-1].weight, gain=1.0)
        if self.critic_head[-1].bias is not None:
            torch.nn.init.constant_(self.critic_head[-1].bias, 0.0)

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

    @staticmethod
    def _detach_hidden(hidden):
        if hidden is None:
            return None
        actor_hidden, critic_hidden = hidden
        return (
            actor_hidden.detach() if actor_hidden is not None else None,
            critic_hidden.detach() if critic_hidden is not None else None,
        )

    def _split_hidden(self, hidden):
        if hidden is None:
            return None, None
        actor_hidden, critic_hidden = hidden
        if actor_hidden is not None:
            actor_hidden = actor_hidden.to(device)
        if critic_hidden is not None:
            critic_hidden = critic_hidden.to(device)
        return actor_hidden, critic_hidden

    def _as_sequence(self, state: torch.Tensor) -> torch.Tensor:
        state = self._sanitize_tensor(state.to(device))
        if state.dim() == 1:
            return state.view(1, 1, -1)
        if state.dim() == 2:
            return state.unsqueeze(0)
        return state

    def _actor_features(self, state_seq: torch.Tensor, hidden=None):
        projected = self.actor_input(state_seq)
        return self.actor_gru(projected, hidden)

    def _critic_features(self, state_seq: torch.Tensor, hidden=None):
        projected = self.critic_input(state_seq)
        return self.critic_gru(projected, hidden)

    def _distribution(self, raw_action: torch.Tensor, learnable_std: bool = True):
        if self.has_continuous_action_space:
            action_mean = self._get_action_mean(raw_action)
            action_mean = self._sanitize_tensor(action_mean)
            log_std = self._get_log_std(learnable=learnable_std)
            std = torch.exp(log_std)
            std_batch = std.expand_as(action_mean)
            scale_tril = torch.diag_embed(std_batch)
            return MultivariateNormal(action_mean, scale_tril=scale_tril)
        return Categorical(logits=raw_action)

    def act(self, state: torch.Tensor, hidden=None):
        state_seq = self._as_sequence(state)
        actor_hidden, critic_hidden = self._split_hidden(hidden)

        actor_out, next_actor_hidden = self._actor_features(state_seq, actor_hidden)
        critic_out, next_critic_hidden = self._critic_features(state_seq, critic_hidden)

        raw = self.actor_head(actor_out[:, -1, :])
        dist = self._distribution(raw, learnable_std=True)
        state_value = self.critic_head(critic_out[:, -1, :])

        action = dist.sample()
        next_hidden = self._detach_hidden((next_actor_hidden, next_critic_hidden))
        return (
            action.squeeze(0).detach(),
            dist.log_prob(action).squeeze(0).detach(),
            state_value.squeeze(0).detach(),
            next_hidden,
        )

    def evaluate(self, state: torch.Tensor, action: torch.Tensor, learnable_std: bool = True):
        logprobs, state_value, dist_entropy, _ = self.evaluate_sequence(
            state,
            action,
            learnable_std=learnable_std,
            predict_next_state=False,
        )
        return logprobs, state_value, dist_entropy

    def evaluate_sequence(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        learnable_std: bool = True,
        predict_next_state: bool = True,
    ):
        state_seq = self._as_sequence(state)
        batch_size, seq_len, _ = state_seq.shape

        actor_out, _ = self._actor_features(state_seq)
        critic_out, _ = self._critic_features(state_seq)
        actor_flat = actor_out.reshape(batch_size * seq_len, -1)
        critic_flat = critic_out.reshape(batch_size * seq_len, -1)

        raw = self.actor_head(actor_flat)
        values = self.critic_head(critic_flat)
        aux_features = torch.cat([actor_flat, critic_flat], dim=-1)
        pred_next_states = self.aux_predictor(aux_features) if predict_next_state else None
        dist = self._distribution(raw, learnable_std=learnable_std)

        flat_action = action.to(device)
        if self.has_continuous_action_space:
            flat_action = flat_action.reshape(batch_size * seq_len, self.action_dim)
        else:
            flat_action = flat_action.reshape(batch_size * seq_len).long()

        logprobs = dist.log_prob(flat_action)
        entropy = dist.entropy()
        return logprobs, values, entropy, pred_next_states

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """外部调用获得状态价值。"""
        state_seq = self._as_sequence(state)
        critic_out, _ = self._critic_features(state_seq)
        return self.critic_head(critic_out[:, -1, :])


class PPO:
    """PPO(Proximal Policy Optimization)算法实现。

    支持离散和连续动作空间,使用裁剪(clipping)的替代目标函数,
    并通过多个 epoch 对同一批数据进行更新。
    """

    def __init__(self, state_dim: int, action_dim: int,
                 lr_actor: float, lr_critic: float,
                 gamma: float, K_epochs: int, eps_clip: float,
                 has_continuous_action_space: bool, action_std_init: float,
                 writer=None, gae_lambda: float = 0.95, aux_loss_coef: float = 0.01,
                 gru_input_dim: int = 128, gru_hidden_dim: int = 256,
                 aux_hidden_dim: int = 128):
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
        self.aux_loss_coef = aux_loss_coef
        self.buffer = RolloutBuffer()           # 经验缓冲区

        # 当前策略网络(用于训练)
        self.policy = ActorCritic(
            state_dim,
            action_dim,
            has_continuous_action_space,
            action_std_init,
            gru_input_dim=gru_input_dim,
            gru_hidden_dim=gru_hidden_dim,
            aux_hidden_dim=aux_hidden_dim,
        ).to(device)
        # 优化器:Actor 和 Critic 使用各自的学习率(通过参数分组实现)
        optimizer_groups = [
            {'params': list(self.policy.actor_parameters()), 'lr': lr_actor},
            {'params': list(self.policy.critic_parameters()), 'lr': lr_critic},
            {'params': list(self.policy.aux_parameters()), 'lr': lr_actor},
        ]
        if has_continuous_action_space:
            optimizer_groups.append({'params': [self.policy.log_std], 'lr': lr_actor})
        self.optimizer = torch.optim.Adam(optimizer_groups)

        # 旧策略网络(用于采样,更新时与当前策略比较)
        self.policy_old = ActorCritic(
            state_dim,
            action_dim,
            has_continuous_action_space,
            action_std_init,
            gru_input_dim=gru_input_dim,
            gru_hidden_dim=gru_hidden_dim,
            aux_hidden_dim=aux_hidden_dim,
        ).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.mse_loss = nn.MSELoss()   # 用于 Critic 的损失函数
        self.update_step = 0
        self.min_update_buffer_size = 128

    @staticmethod
    def _sanitize_tensor(t: torch.Tensor) -> torch.Tensor:
        """清理 PPO 入口张量中的 NaN/Inf。"""
        return torch.where(
            torch.isnan(t) | torch.isinf(t), torch.zeros_like(t), t
        )

    def select_action(self, state, hidden=None):
        state = self._sanitize_tensor(state.to(device))
        with torch.no_grad():
            action, logprob, state_value, next_hidden = self.policy_old.act(state, hidden)
            state_value = state_value.squeeze(-1)

        self.buffer.states.append(state.cpu().numpy().tolist())
        self.buffer.actions.append(action.cpu().numpy().tolist())
        self.buffer.logprobs.append(logprob.cpu().numpy().tolist())
        self.buffer.state_values.append(float(state_value.item()))

        action_value = action.cpu().numpy() if self.has_continuous_action_space else action.item()
        return action_value, next_hidden

    @staticmethod
    def _build_segments(terminals: torch.Tensor) -> list[tuple[int, int]]:
        segments = []
        start = 0
        for idx, terminal in enumerate(terminals.tolist()):
            if terminal:
                if idx + 1 - start >= 2:
                    segments.append((start, idx + 1))
                start = idx + 1
        if terminals.numel() - start >= 2:
            segments.append((start, terminals.numel()))
        return segments

    @staticmethod
    def _state_prediction_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean((pred - target) ** 2)

    def update(self):
        """使用缓冲区中收集的经验更新策略网络(PPO 核心更新步骤)。"""
        if not self.buffer.rewards:
            return None
        if len(self.buffer.rewards) < self.min_update_buffer_size:
            return None
        if len(self.buffer.next_states) != len(self.buffer.states):
            logger.warning("PPO update skipped because next_states length does not match states")
            self.buffer.clear()
            return None

        # ========== 1. 将缓冲区中的列表数据转换为张量 ==========
        # 状态:从列表转换并堆叠
        old_states = torch.tensor(self.buffer.states, dtype=torch.float32, device=device).detach()
        old_next_states = torch.tensor(self.buffer.next_states, dtype=torch.float32, device=device).detach()
        # 动作
        action_dtype = torch.float32 if self.has_continuous_action_space else torch.long
        old_actions = torch.tensor(self.buffer.actions, dtype=action_dtype, device=device).detach()
        if self.has_continuous_action_space and old_actions.dim() == 1:
            old_actions = old_actions.unsqueeze(-1)
        # 旧对数概率
        old_logprobs = torch.tensor(self.buffer.logprobs, dtype=torch.float32, device=device).detach()
        old_logprobs = old_logprobs.reshape(-1)
        old_state_values = torch.tensor(self.buffer.state_values, dtype=torch.float32, device=device).detach()

        # ========== 2. 使用真实 GAE 计算优势和 returns ==========
        rewards = torch.tensor(self.buffer.rewards, dtype=torch.float32, device=device)
        terminals_bool = torch.tensor(self.buffer.is_terminals, dtype=torch.bool, device=device)
        terminals = terminals_bool.float()

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
        segments = self._build_segments(terminals_bool)
        if not segments:
            logger.warning("PPO update skipped because no recurrent segment has length >= 2")
            self.buffer.clear()
            return None

        # ========== 3. 多次 epoch 更新策略 ==========
        actor_loss_value = 0.0
        critic_loss_value = 0.0
        aux_loss_value = 0.0
        total_loss_value = 0.0
        entropy_value = 0.0
        grad_norm_value = 0.0
        for _ in range(self.K_epochs):
            logprob_parts = []
            value_parts = []
            entropy_parts = []
            aux_loss_parts = []

            for start, end in segments:
                seg_states = old_states[start:end]
                seg_actions = old_actions[start:end]
                seg_next_states = old_next_states[start:end]
                logprobs, state_values, dist_entropy, pred_next_states = self.policy.evaluate_sequence(
                    seg_states,
                    seg_actions,
                    predict_next_state=True,
                )
                logprob_parts.append(logprobs)
                value_parts.append(state_values.squeeze(-1))
                entropy_parts.append(dist_entropy)
                aux_loss_parts.append(self._state_prediction_mse(pred_next_states, seg_next_states))

            logprobs = torch.cat(logprob_parts, dim=0)
            state_values = torch.cat(value_parts, dim=0)
            dist_entropy = torch.cat(entropy_parts, dim=0)
            segment_indices = torch.cat([
                torch.arange(start, end, device=device) for start, end in segments
            ])

            seg_old_logprobs = old_logprobs[segment_indices]
            seg_advantages = advantages[segment_indices]
            seg_returns = returns[segment_indices]

            # 计算概率比 r(θ) = π_θ(a|s) / π_old(a|s)
            ratios = torch.exp(logprobs - seg_old_logprobs.detach())

            # PPO 裁剪损失
            surr1 = ratios * seg_advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * seg_advantages
            actor_loss = -torch.min(surr1, surr2)
            critic_loss = self.mse_loss(state_values, seg_returns)
            entropy_bonus = dist_entropy
            aux_loss = torch.stack(aux_loss_parts).mean()
            loss = actor_loss.mean() + 0.5 * critic_loss - 0.01 * entropy_bonus.mean()
            loss = loss + self.aux_loss_coef * aux_loss

            if not torch.isfinite(loss):
                logger.warning("PPO update skipped because loss contains NaN/Inf")
                self.buffer.clear()
                return None

            actor_loss_value = actor_loss.mean().item()
            critic_loss_value = critic_loss.item()
            aux_loss_value = aux_loss.item()
            total_loss_value = loss.item()
            entropy_value = entropy_bonus.mean().item()

            # 反向传播与参数更新
            self.optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪防止梯度爆炸
            grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            grad_norm_value = float(grad_norm.item())
            self.optimizer.step()

        self.update_step += 1

        # 更新完成后,将旧策略网络同步为当前策略网络
        self.policy_old.load_state_dict(self.policy.state_dict())
        # 清空缓冲区,准备下一轮收集
        self.buffer.clear()
        return {
            "actor_loss": actor_loss_value,
            "critic_loss": critic_loss_value,
            "aux_loss": aux_loss_value,
            "total_loss": total_loss_value,
            "entropy": entropy_value,
            "buffer_size": len(rewards),
            "grad_norm": grad_norm_value,
            "sequence_count": len(segments),
            "mean_sequence_len": sum(end - start for start, end in segments) / len(segments),
        }

    def load(self, checkpoint_path: str):
        """从文件加载模型权重(同时更新 policy 和 policy_old)。"""
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=device))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=device))

    def save(self, checkpoint_path: str):
        """将当前策略网络的权重保存到文件。"""
        torch.save(self.policy.state_dict(), checkpoint_path)
