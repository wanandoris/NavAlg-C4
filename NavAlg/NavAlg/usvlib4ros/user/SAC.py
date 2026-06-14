import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import tyro
import torch.nn as nn
import numpy as np
import time
from collections import namedtuple
from dataclasses import dataclass

#===========================================记得改=================================

ACTION_MAX =  np.array([1,2], dtype=np.float32)   #动作最大数值，它和min都是多维度numpy数据，对应不同的action====根据nav,这里两维度是speed==rotate
ACTION_MIN = np.array([0,-2], dtype=np.float32)   #在nav里会把action*100，因此这里仅仅把范围限制在（0,1）和（-2,2）
OBS_SPACE = (94,)    #======================修改294行self.obs
ACTION_SPACE = (2, )

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    torch_deterministic: bool = True
    cuda: bool = True
    total_timesteps: int = 1000000
    num_envs: int = 1
    buffer_size: int = int(512)
    gamma: float = 0.98      #根据仿真看到未来50步就差不多的情况调整的
    tau: float = 0.02    #============参数按多少权重向新的target网络更新
    batch_size: int = 256
    policy_lr: float = 9e-5
    q_lr: float = 4e-4
    policy_frequency: int = 2     #======================执行几次critic更新后再更新policy
    target_network_frequency: int = 1  # =======目标网络更新频率，不用管
    alpha: float = 0.2    #=============探索能力
    autotune: bool = True  #====================是否自动调alpha
    learning_starts: int = buffer_size
    save: int = 1024      #===========多少步保存
    file: str = "D:\\大赛资源\\智能导航C4-2026\\unpack\\NavAlg-C4-v1\\model"
    learn_step : int = 512    #=============每多少步learn一次


Transition = namedtuple(
    'Transition',
    (
        'observations',
        'actions',
        'rewards',
        'next_observations',
        'dones'
    )
)


class ReplayBuffer:
    def __init__(
        self,
        buffer_size,
        obs_space,
        action_space,
        device,
        envs
    ):
        self.buffer_size = buffer_size
        self.device = device
        self.n_envs = envs

        self.obs_shape = obs_space

        self.action_dim = action_space[0]
            

        # (time, env, ...)
        self.observations = np.zeros(
            (buffer_size, envs, *self.obs_shape),
            dtype=np.float32
        )

        self.actions = np.zeros(
            (buffer_size, envs, self.action_dim),
            dtype=np.float32
        )

        self.rewards = np.zeros(
            (buffer_size, envs),
            dtype=np.float32
        )

        self.next_observations = np.zeros(
            (buffer_size, envs, *self.obs_shape),
            dtype=np.float32
        )

        self.terminations = np.zeros(
            (buffer_size, envs),
            dtype=np.float32
        )

        self.truncations = np.zeros(
            (buffer_size, envs),
            dtype=np.float32
        )

        self.pos = 0
        self.size = 0

    def add(
        self,
        obs,
        action,
        reward,
        next_obs,
        terminated
    ):
        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_observations[self.pos] = next_obs
        self.terminations[self.pos] = terminated

        self.pos = (self.pos + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size):
        """
        返回：

        observations      (batch_size*n_envs, *obs_shape)
        actions           (batch_size*n_envs, action_dim)
        rewards           (batch_size*n_envs,)
        next_observations (batch_size*n_envs, *obs_shape)
        dones             (batch_size*n_envs,)
        """

        indices = np.random.randint(      #============随机sample
            0,
            self.size,
            size=batch_size
        )

        obs = self.observations[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        next_obs = self.next_observations[indices]

        dones = np.logical_or(
            self.terminations[indices],0
        ).astype(np.float32)

        # ------------------------------------------------
        # 展平 env 维度
        # (B, N, ...)
        # -> (B*N, ...)
        # ------------------------------------------------

        obs = obs.reshape(
            batch_size * self.n_envs,
            *self.obs_shape
        )

        next_obs = next_obs.reshape(
            batch_size * self.n_envs,
            *self.obs_shape
        )

        actions = actions.reshape(
            batch_size * self.n_envs,
            self.action_dim
        )

        rewards = rewards.reshape(
            batch_size * self.n_envs
        )

        dones = dones.reshape(
            batch_size * self.n_envs
        )

        return Transition(
            observations=torch.as_tensor(
                obs,
                dtype=torch.float32,
                device=self.device
            ),

            actions=torch.as_tensor(
                actions,
                dtype=torch.float32,
                device=self.device
            ),

            rewards=torch.as_tensor(
                rewards,
                dtype=torch.float32,
                device=self.device
            ),

            next_observations=torch.as_tensor(
                next_obs,
                dtype=torch.float32,
                device=self.device
            ),

            dones=torch.as_tensor(
                dones,
                dtype=torch.float32,
                device=self.device
            ),
        )

    def __len__(self):
        return self.size





class SoftQNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(
            np.prod(OBS_SPACE) + np.prod(ACTION_SPACE),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(np.array(OBS_SPACE).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(ACTION_SPACE))
        self.fc_logstd = nn.Linear(256, np.prod(ACTION_SPACE))     #与mean独立，从而学到不同的网络
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (ACTION_MAX - ACTION_MIN) / 2.0,    #缩放系数,把输出映射为以0为中心的，范围为min~max的数值。
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",                          #之前是以0为原点的动作分布，这里改为正确的分布
            torch.tensor(
                (ACTION_MAX + ACTION_MIN) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)   #构建正态分布对象，来随机取a
        x_t = normal.rsample()  # 比普通sample多了梯度以便之后计算
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias           #真正取动作
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean



class SAC:
    def __init__(self,ifload,file_path = r"D:\大赛资源\智能导航C4-2026\unpack\NavAlg-C4-v1\model"):
        """
        传入ifloaad,保存的pt文件路径,reset的obs
        路径参考：    file: str = r"D:\\大赛资源\\智能导航C4-2026\\unpack\\NavAlg\\models"
        """

        self.args = tyro.cli(Args)
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.args.cuda else "cpu")

        self.obs = np.zeros((1,94))
        self.actor = Actor().to(self.device)
        self.qf1 = SoftQNetwork().to(self.device)
        self.qf2 = SoftQNetwork( ).to(self.device)
        self.qf1_target = SoftQNetwork(    ).to(self.device)
        self.qf2_target = SoftQNetwork(    ).to(self.device)
        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=self.args.q_lr)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=self.args.policy_lr)
        self.learn_time : int = 0   #记录训练次数
        
        if self.args.autotune:                        #alpha不断更新
            self.target_entropy = -torch.prod(torch.Tensor(ACTION_SPACE).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=self.args.q_lr)
        else:
            self.alpha = self.args.alpha
            
        if ifload:                                  #保存
            self.load(filepath= file_path)
        else:
            self.qf1_target.load_state_dict(self.qf1.state_dict())
            self.qf2_target.load_state_dict(self.qf2.state_dict())
            
        self.rb = ReplayBuffer(
        self.args.buffer_size,
        OBS_SPACE,
        ACTION_SPACE,
        self.device,
        envs=self.args.num_envs
        
        )
        

        
    def run(self,next_obs, rewards, terminations,global_step,episode_r):
        
        """
        需要：
        obs.shape == (1,184)      非torch即可
        next_obs.shape == (1,184)  同上
        
        obs要求np，不过好像本来就是
        返回action
        action未消去第0维
        """

        if global_step < self.args.learning_starts:
            actions = np.array([np.random.uniform(low=ACTION_MIN, high=ACTION_MAX) for _ in range(self.args.num_envs)])
        else:
            actions, _, _ = self.actor.get_action(torch.Tensor(self.obs).to(self.device))
            actions = actions.detach().cpu().numpy()
            
        real_next_obs = next_obs.copy()
        self.rb.add(self.obs, actions, rewards, real_next_obs, terminations)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        self.obs = next_obs
        if global_step >=  self.args.learning_starts:
            if global_step % self.args.learn_step ==0:
                self.learn(global_step)
        if global_step % self.args.save == 0:
            self.save(self.args.file,global_step,episode_r)
        
        print("==训练次数",self.learn_time)
        return actions 
        
    def learn(self,global_step):
            data = self.rb.sample(self.args.batch_size)
            self.learn_time += 1
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = self.actor.get_action(data.next_observations)
                qf1_next_target = self.qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = self.qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * self.args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = self.qf1(data.observations, data.actions).view(-1)
            qf2_a_values = self.qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss
            
            # optimize the model
            self.q_optimizer.zero_grad()
            qf_loss.backward()
            self.q_optimizer.step()

            if global_step % self.args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    self.args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = self.actor.get_action(data.observations)
                    qf1_pi = self.qf1(data.observations, pi)
                    qf2_pi = self.qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    if self.args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = self.actor.get_action(data.observations)
                        alpha_loss = (-self.log_alpha.exp() * (log_pi + self.target_entropy)).mean()

                        self.a_optimizer.zero_grad()
                        alpha_loss.backward()
                        self.a_optimizer.step()
                        self.alpha = self.log_alpha.exp().item()

            # update the target networks
            if global_step % self.args.target_network_frequency == 0:
                for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                    target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)
                for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                    target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)
    
    def save(self, filepath: str, total_steps, episode_r):
        """保存 SAC 完整训练状态"""
        filepath = os.path.join(filepath, "sac_%d_reward_%d.pt"%(total_steps,episode_r))

        checkpoint = {
            "actor": self.actor.state_dict(),
            "qf1": self.qf1.state_dict(),
            "qf2": self.qf2.state_dict(),
            "qf1_target": self.qf1_target.state_dict(),
            "qf2_target": self.qf2_target.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "total_steps": total_steps,
        }
        # autotune 相关状态
        if self.args.autotune:
            checkpoint["log_alpha"] = self.log_alpha.detach().cpu()
            checkpoint["a_optimizer"] = self.a_optimizer.state_dict()

        torch.save(checkpoint, filepath)
        print(f"[SAC] Saved to {filepath} (steps={total_steps})")

    def load(self, filepath: str = None):
        """
        恢复 SAC 完整训练状态。
        注意：此方法假设 self._create_networks() 和 self._create_optimizers() 
        已在 __init__ 中按正确顺序调用完毕。
        """
        if filepath is None:
            # 根据 run_name 或其他规则自动定位最新 checkpoint
            raise ValueError("请提供 checkpoint 路径，或在类中实现自动查找逻辑")
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)

        # 加载网络权重（Target 网络直接从 checkpoint 恢复，不被覆盖）
        self.actor.load_state_dict(checkpoint["actor"])
        self.qf1.load_state_dict(checkpoint["qf1"])
        self.qf2.load_state_dict(checkpoint["qf2"])
        self.qf1_target.load_state_dict(checkpoint["qf1_target"])
        self.qf2_target.load_state_dict(checkpoint["qf2_target"])

        # 加载优化器状态（动量、自适应学习率等）
        self.q_optimizer.load_state_dict(checkpoint["q_optimizer"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        if self.args.autotune:
            self.log_alpha.data.copy_(checkpoint["log_alpha"])
            self.a_optimizer.load_state_dict(
                checkpoint["a_optimizer"]
            )
    
