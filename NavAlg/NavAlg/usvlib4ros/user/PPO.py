import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from collections import namedtuple

import pandas as pd
import os

ACTION_MAX =  np.array([1,2], dtype=np.float32)   #动作最大数值，它和min都是多维度numpy数据，对应不同的action====根据nav,这里两维度是speed==rotate
ACTION_MIN = np.array([0,-2], dtype=np.float32)   #在nav里会把action*100，因此这里仅仅把范围限制在（0,1）和（-2,2）
OBS_DIM = 184
ACTION_DIM = 2

OBS_SPACE = (OBS_DIM + ACTION_DIM,)
ACTION_SPACE = (ACTION_DIM,)

IF_LEARN = True

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




@dataclass
class Args:
    cuda: bool = True
    total_timesteps: int = 1000000
    learning_rate: float = 9e-4
    num_envs: int = 1
    num_steps: int = 512       #每次策略更新收集的数据
    anneal_lr: bool = True   
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 8    #一批数据几次梯度更新
    update_epochs: int = 3
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.001
    vf_coef: float = 0.5
    max_grad_norm: float = 2
    target_kl: float = None

    batch_size: int = num_steps
    minibatch_size: int = 0    #初始化时赋值，这里不用
    num_iterations: int = 0    #虽然不在这里决定，但决定了anneal_lr的变化。让它与nav保持一致，或者把anneal_lr改为依赖于奖励
    save : int = 1024
    
    file :str =  "D:\\大赛资源\\智能导航C4-2026\\unpack\\NavAlg-C4-v1\\ppo_models"

    
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class MODEL(nn.Module):
    def __init__(self):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(OBS_SPACE).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(OBS_SPACE).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(ACTION_SPACE)), std=0.01),
        )
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
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(ACTION_SPACE)))



    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_ = None
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            x_t = probs.rsample()  # 比普通sample多了梯度以便之后计算
            y_t = torch.tanh(x_t)
            action_ = y_t * self.action_scale + self.action_bias           #真正取动作
            action = x_t.detach()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x),action_

    
class PPO:
    def __init__(self,ifload,file_path = r"D:\大赛资源\智能导航C4-2026\unpack\NavAlg-C4-v1\model"):
        
        self.args = tyro.cli(Args)
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.args.cuda else "cpu")
        self.args.minibatch_size = int(self.args.batch_size // self.args.num_minibatches)
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.args.cuda else "cpu")

        self.agent = MODEL().to(self.device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.args.learning_rate, eps=1e-5)
        self.obs = torch.zeros((self.args.num_steps, self.args.num_envs) + OBS_SPACE).to(self.device)
        self.actions = torch.zeros((self.args.num_steps, self.args.num_envs) + ACTION_SPACE).to(self.device)
        self.logprobs = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        self.rewards = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        self.dones = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        self.values = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        self.action = torch.zeros((2,)).to(self.device)
        self.per_goal = 1024    #每1024步多少次goal
        
        self.episode_r_list = []    #存储所有episoder，来判断训练效果。若效果变好，则降低lr
        self.dict = {
            "learn_time": 0,
            "arrive_time": 0,
            "reward": 0.0,
            "explain_var": 0.0
        }
        self.learntime = 0
        self.arrive_time = 0
        
        if ifload:                                  #保存
            self.load(self.agent,filepath= file_path)
        


    def log(self, r_sum, explain_var,success_rate):

        self.dict["learn_time"] = self.learntime
        self.dict["success_rate"] = success_rate
        self.dict["reward"] = r_sum
        self.dict["explain_var"] = explain_var

        file_path = r"D:\大赛资源\智能导航C4-2026\unpack\NavAlg-C4-v1\training_log.csv"

        new_row = pd.DataFrame([self.dict])

        if os.path.exists(file_path):
            new_row.to_csv(
                file_path,
                mode='a',
                header=False,
                index=False
            )
        else:
            new_row.to_csv(
                file_path,
                index=False
            )
        return True

        

    def learn(self,next_obs,next_done,success_rate):
        self.learntime += 1
        
        r_sum = sum(self.rewards)
        
        
        with torch.no_grad():
            next_value = self.agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(self.rewards).to(self.device)
            lastgaelam = 0
            for t in reversed(range(self.args.num_steps)):
                if t == self.args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.dones[t + 1]
                    nextvalues = self.values[t + 1]
                delta = self.rewards[t] + self.args.gamma * nextvalues * nextnonterminal - self.values[t]
                advantages[t] = lastgaelam = delta + self.args.gamma * self.args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + self.values

        # flatten the batch
        b_obs = self.obs.reshape((-1,) + OBS_SPACE)
        b_logprobs = self.logprobs.reshape(-1)
        b_actions = self.actions.reshape((-1,) + ACTION_SPACE)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(self.args.batch_size)
        clipfracs = []
        for epoch in range(self.args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.args.batch_size, self.args.minibatch_size):
                end = start + self.args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue,_ =self.agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if self.args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if self.args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.args.clip_coef,
                        self.args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.args.ent_coef * entropy_loss + v_loss * self.args.vf_coef
                if IF_LEARN:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.args.max_grad_norm)
                    self.optimizer.step()

            if self.args.target_kl is not None and approx_kl > self.args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        islog = False
        islog = self.log(r_sum,explained_var,success_rate)
        self.arrive_time = 0
        return islog
        
    def save(self,model, filepath,total_steps,episode_r,done_time):
        """
        保存 PPO 模型的检查点
        """
        filepath = os.path.join(filepath, "ppo_%d.pt"%(self.learntime))
        checkpoint = {
            'model_state_dict': model.state_dict(),          # 保存 actor 和 critic 的网络权重
            'optimizer_state_dict': self.optimizer.state_dict(), # 保存优化器状态（如 Adam 的动量等）
        }
        torch.save(checkpoint, filepath)
        print(f"✅ Checkpoint saved successfully to {filepath}")
        
    def load(self,model, filepath):
        """
        加载 PPO 模型的检查点
        """
        checkpoint = torch.load(filepath, map_location=self.device,weights_only=False) # 自动适配 CPU/GPU
        
        model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"🔄 Checkpoint loaded successfully from {filepath} ")


    def anneal_lr(self,episode_r):
        self.episode_r_list.append(episode_r)
        if self.args.anneal_lr and len(self.episode_r_list)>10:
            frac = self.sigmoid_v1(1.0 - (episode_r - max(self.episode_r_list)) / max(self.episode_r_list))
            lrnow = frac * self.args.learning_rate
            print("==================lrnow",lrnow)
            self.optimizer.param_groups[0]["lr"] = lrnow
        if len(self.episode_r_list) > 100: 
            self.episode_r_list.pop(0)

    def sigmoid_v1(self,x):
        """
        x轴上收缩了4倍，以保证大概在x=1时y趋近于1，而x再大时y差不多不变了
        对于结果进行处理，最后为-1~1
        Sigmoid 激活函数（包含数值稳定性优化）
        :param x: 输入值（可以是标量、列表或 NumPy 数组）
        :return: 经过 Sigmoid 处理后的输出，范围在 (0, 1) 之间
        """
        result = 1 / (1 + np.exp(-x*4))
        return (result-0.5)*2
    
    def run(self,next_obs, reward, terminations,global_step,episode_r,success_rate,heading_diff):
        
        step = global_step % self.args.num_steps
        next_done = np.logical_or(terminations,0).astype(int)
        self.rewards[step] = torch.tensor(reward).to(self.device).view(-1)
        next_obs, next_done = torch.tensor(next_obs,dtype=torch.float32).to(self.device), torch.tensor(next_done).to(self.device)
        self.action[1] = heading_diff
        next_obs = torch.cat([next_obs,self.action],dim=0).unsqueeze(0) 
        self.dones[step] = next_done
        with torch.no_grad():
                action, logprob, _, value,action_ = self.agent.get_action_and_value(next_obs)
                self.values[step] = value.flatten()
        self.actions[step] = action
        self.action = action.squeeze(0)
        self.obs[step] = next_obs
        self.logprobs[step] = logprob
        print("==========learntime",self.learntime)
        islog = False
        if step == 0 and global_step > 0:
            islog = self.learn(next_obs,next_done,success_rate)
        if global_step % self.args.save == 0:
            self.save(self.agent,self.args.file,global_step,episode_r,success_rate)
        return action_.cpu().numpy(),islog