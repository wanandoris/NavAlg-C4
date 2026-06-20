import logging
import math

import torch
import torch.nn as nn
from torch.distributions import Categorical, MultivariateNormal

logger = logging.getLogger(__name__)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class RolloutBuffer:
    """PPO 轨迹缓冲区。"""

    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.state_values = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
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

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.log_std = nn.Parameter(
                torch.full((action_dim,), math.log(action_std_init), dtype=torch.float32)
            )

        self._init_weights()

    def _init_weights(self):
        for module in self.shared.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0.0)

        torch.nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        if self.actor_mean.bias is not None:
            torch.nn.init.constant_(self.actor_mean.bias, 0.0)

        torch.nn.init.orthogonal_(self.critic.weight, gain=1.0)
        if self.critic.bias is not None:
            torch.nn.init.constant_(self.critic.bias, 0.0)

    @staticmethod
    def _sanitize_tensor(t: torch.Tensor) -> torch.Tensor:
        return torch.where(torch.isnan(t) | torch.isinf(t), torch.zeros_like(t), t)

    def _get_log_std(self, learnable: bool) -> torch.Tensor:
        log_std = torch.clamp(self.log_std, min=-3.0, max=-0.3)
        return log_std if learnable else log_std.detach()

    def _get_action_mean(self, raw_action: torch.Tensor) -> torch.Tensor:
        raw_action = self._sanitize_tensor(raw_action)
        return torch.tanh(raw_action)

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
            action_mean = self._sanitize_tensor(self._get_action_mean(raw))
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
        state = self._sanitize_tensor(state.to(device))
        return self.critic(self.shared(state))


class PPO:
    """PPO(Proximal Policy Optimization)算法实现。"""

    def __init__(self, state_dim: int, action_dim: int,
                 lr_actor: float, lr_critic: float,
                 gamma: float, K_epochs: int, eps_clip: float,
                 has_continuous_action_space: bool, action_std_init: float,
                 writer=None, gae_lambda: float = 0.95):
        self.has_continuous_action_space = has_continuous_action_space
        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        params = [
            {'params': self.policy.shared.parameters(), 'lr': lr_actor},
            {'params': self.policy.actor_mean.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic},
        ]
        if has_continuous_action_space:
            params.insert(2, {'params': [self.policy.log_std], 'lr': lr_actor})
        self.optimizer = torch.optim.Adam(params)

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.mse_loss = nn.MSELoss()
        self.tb_writer = writer
        self.update_step = 0
        self.min_update_buffer_size = 128

    @staticmethod
    def _sanitize_tensor(t: torch.Tensor) -> torch.Tensor:
        return torch.where(torch.isnan(t) | torch.isinf(t), torch.zeros_like(t), t)

    def select_action(self, state):
        state = self._sanitize_tensor(state.to(device))
        with torch.no_grad():
            action, logprob = self.policy_old.act(state)
            state_value = self.policy_old.get_value(state).squeeze(-1)

        self.buffer.states.append(state.cpu().numpy().tolist())
        self.buffer.actions.append(action.cpu().numpy().tolist())
        self.buffer.logprobs.append(logprob.cpu().numpy().tolist())
        self.buffer.state_values.append(float(state_value.item()))
        return action.cpu().numpy() if self.has_continuous_action_space else action.item()

    def update(self):
        if not self.buffer.rewards:
            return None
        if len(self.buffer.rewards) < self.min_update_buffer_size:
            return None

        old_states = torch.tensor(self.buffer.states, dtype=torch.float32, device=device).detach()
        old_actions = torch.tensor(self.buffer.actions, dtype=torch.float32, device=device).detach()
        if self.has_continuous_action_space and old_actions.dim() == 1:
            old_actions = old_actions.unsqueeze(-1)
        old_logprobs = torch.tensor(self.buffer.logprobs, dtype=torch.float32, device=device).detach()
        old_state_values = torch.tensor(self.buffer.state_values, dtype=torch.float32, device=device).detach()

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

        actor_loss_value = 0.0
        critic_loss_value = 0.0
        total_loss_value = 0.0
        entropy_value = 0.0
        ratio_mean_value = 0.0
        advantage_mean_value = advantages.mean().item()

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = state_values.squeeze(-1)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2)
            critic_loss = self.mse_loss(state_values, returns)
            entropy_bonus = dist_entropy
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

            self.optimizer.zero_grad()
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            self.optimizer.step()

        if self.tb_writer is not None:
            try:
                self.tb_writer.add_scalar("ppo/total_loss", total_loss_value, self.update_step)
                self.tb_writer.add_scalar("ppo/actor_loss", actor_loss_value, self.update_step)
                self.tb_writer.add_scalar("ppo/critic_loss", critic_loss_value, self.update_step)
                self.tb_writer.add_scalar("ppo/entropy", entropy_value, self.update_step)
                self.tb_writer.add_scalar("ppo/ratio_mean", ratio_mean_value, self.update_step)
                self.tb_writer.add_scalar("ppo/advantage_mean", advantage_mean_value, self.update_step)
            except Exception:
                pass
        self.update_step += 1

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()
        return {
            "actor_loss": actor_loss_value,
            "critic_loss": critic_loss_value,
            "total_loss": total_loss_value,
            "entropy": entropy_value,
            "buffer_size": len(rewards),
        }

    def load(self, checkpoint_path: str):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=device))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=device))

    def save(self, checkpoint_path: str):
        torch.save(self.policy.state_dict(), checkpoint_path)