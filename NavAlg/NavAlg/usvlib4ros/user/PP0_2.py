import logging
import torch
import torch.nn as nn
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
    """Actor-Critic网络:Actor输出策略分布,Critic估计状态值。
    
    支持离散动作空间(使用Categorical分布)和连续动作空间(使用MultivariateNormal分布)。
    """

    def __init__(self, state_dim: int, action_dim: int,
                 has_continuous_action_space: bool, action_std_init: float):
        """初始化 Actor 和 Critic 网络结构。

        Args:
            state_dim: 状态空间的维度
            action_dim: 动作空间的维度(离散时为动作数,连续时为动作向量的维度)
            has_continuous_action_space: 是否为连续动作空间
            action_std_init: 连续动作空间下,初始化标准差(用于构建方差对角矩阵)
        """
        super().__init__()
        # 是否有连续动作空间
        self.has_continuous_action_space = has_continuous_action_space

        #如果是连续动作空间,初始化动作方差
        if has_continuous_action_space:
            self.action_dim = action_dim
            # 学习 log_std 比直接学习方差更稳定,并可保证 std 始终为正
            self.log_std = nn.Parameter(
                torch.full((action_dim,), math.log(action_std_init), dtype=torch.float32)
            )

        # ========== Actor 网络 ==========
        # 输入状态 -> 输出动作的概率(离散)或动作均值(连续)
        if not has_continuous_action_space:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 256),   # 第一层全连接
                nn.Tanh(),                   # 激活函数
                nn.Linear(256, 256),         # 第二层全连接
                nn.Tanh(),
                nn.Linear(256, action_dim),  # 输出层
                # 如果是离散动作空间,最后添加 Softmax 转换为概率分布；
                nn.Softmax(dim=-1)
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 256),   # 第一层全连接
                nn.Tanh(),                   # 激活函数
                nn.Linear(256, 256),         # 第二层全连接
                nn.Tanh(),
                nn.Linear(256, action_dim),  # 输出层
            )

        # ========== Critic 网络 ==========
        # 输入状态 -> 输出状态价值 V(s) (标量)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

    @staticmethod
    def _sanitize_tensor(t: torch.Tensor) -> torch.Tensor:
        """将张量中的 NaN 或 Inf 替换为 0,防止数值不稳定导致训练失败。

        Args:
            t: 输入张量

        Returns:
            处理后的张量,无效值被置为0
        """
        return torch.where(
            torch.isnan(t) | torch.isinf(t), torch.zeros_like(t), t
        )

    def _get_log_std(self, learnable: bool) -> torch.Tensor:
        """在启用前冻结 log_std，首次到达后再允许其参与梯度更新。"""
        log_std = torch.clamp(self.log_std, min=-2.5, max=0.5)
        return log_std if learnable else log_std.detach()

    def act(self, state: torch.Tensor):
        """根据当前状态采样一个动作,并返回动作及其对数概率(不计算梯度)。

        Args:
            state: 状态张量,形状通常为 (state_dim,) 或 (batch, state_dim)

        Returns:
            action: 采样的动作(张量)
            log_prob: 该动作对应的对数概率(张量)
        """
        state = self._sanitize_tensor(state.to(device))

        if self.has_continuous_action_space:
            # 连续动作空间:Actor 输出动作均值,使用对角协方差矩阵构建正态分布
            raw = self.actor(state)
            action_mean = self._get_action_mean(raw)
            log_std = self._get_log_std(learnable=True)
            std = torch.exp(log_std)  # 标准差向量
            scale_tril = torch.diag(std)
            dist = MultivariateNormal(action_mean, scale_tril=scale_tril)

        else:
            # 离散动作空间:Actor 输出概率分布,需限制数值范围避免极值
            action_probs = torch.clamp(self.actor(state), min=1e-6, max=1 - 1e-6)
            action_probs = self._sanitize_tensor(action_probs)
            # 归一化确保概率之和为 1(防止数值误差导致和不为 1)
            action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
            dist = Categorical(action_probs)

        action = dist.sample()
        return action.detach(), dist.log_prob(action).detach()

    def evaluate(self, state: torch.Tensor, action: torch.Tensor, learnable_std: bool = True):
        """评估给定状态-动作对的对数概率、状态价值以及分布的熵。

        用于 PPO 更新阶段,根据旧策略收集的数据计算新策略的相关量。

        Args:
            state: 状态张量
            action: 实际执行的动作张量

        Returns:
            log_prob: 在当前策略下,该状态-动作对的对数概率
            state_value: Critic 网络给出的状态价值 V(s)
                : 策略分布的熵,用于鼓励探索
        """
        state = self._sanitize_tensor(state.to(device))

        if self.has_continuous_action_space:
            raw = self._sanitize_tensor(self.actor(state))
            action_mean = self._get_action_mean(raw)
            action_mean = self._sanitize_tensor(action_mean)
            log_std = self._get_log_std(learnable=learnable_std)
            std = torch.exp(log_std)                         # 标准差向量
            # 扩展到批次维度：将 (action_dim,) 复制到 (batch, action_dim)
            std_batch = std.unsqueeze(0).expand(action_mean.size(0), -1)
            scale_tril = torch.diag_embed(std_batch)          # 批量对角矩阵
            dist = MultivariateNormal(action_mean, scale_tril=scale_tril)
        else:
            action_probs = torch.clamp(self.actor(state), min=1e-6, max=1 - 1e-6)
            action_probs = self._sanitize_tensor(action_probs)
            action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
            dist = Categorical(action_probs)

        return dist.log_prob(action), self.critic(state), dist.entropy()
    
    def _get_action_mean(self, raw_action):
        """对 Actor 输出的原始值分别施加不同的激活函数，得到动作均值。"""
        # 避免任何原地操作，防止破坏 Sigmoid/Tanh 的反向传播图
        raw_action = self._sanitize_tensor(raw_action)
        mean_0 = torch.tanh(raw_action[..., 0:1])
        mean_1 = torch.sigmoid(raw_action[..., 1:2])
        return torch.cat([mean_0, mean_1], dim=-1)


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
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params' : [self.policy.log_std],'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        # 旧策略网络(用于采样,更新时与当前策略比较)
        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.mse_loss = nn.MSELoss()   # 用于 Critic 的损失函数
        self.tb_writer = TensorBoardMetricsWriter(writer=writer) if writer is not None else None
        self.update_step = 0

    @staticmethod
    def _sanitize_tensor(t: torch.Tensor) -> torch.Tensor:
        """清理 PPO 入口张量中的 NaN/Inf。"""
        return torch.where(
            torch.isnan(t) | torch.isinf(t), torch.zeros_like(t), t
        )

    def select_action(self, state) -> int:
        """根据当前状态选择动作,并将经验存入缓冲区。

        Args:
            state: 当前状态(可以是 numpy 数组或 torch 张量)

        Returns:
            action: 选出的动作(标量 int,用于离散动作空间)
        """
        state = self._sanitize_tensor(state.to(device))                     # 确保状态在正确的设备上
        with torch.no_grad():                        # 采样时无需梯度
            action, logprob = self.policy_old.act(state)
            state_value = self.policy_old.critic(state).squeeze(-1)

        # 将状态、动作、对数概率存储到缓冲区(转为 Python 列表以方便后续 JSON 序列化)
        self.buffer.states.append(state.cpu().numpy().tolist())
        self.buffer.actions.append(action.cpu().numpy().tolist())
        self.buffer.logprobs.append(logprob.cpu().numpy().tolist())
        self.buffer.state_values.append(float(state_value.item()))

        return action.cpu().numpy() if self.has_continuous_action_space else action.item()  # 返回标量动作值

    def update(self):
        """使用缓冲区中收集的经验更新策略网络(PPO 核心更新步骤)。"""
        if not self.buffer.rewards:
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
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
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
