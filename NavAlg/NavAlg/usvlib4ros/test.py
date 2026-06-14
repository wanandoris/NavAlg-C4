import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    """
    Sigmoid 激活函数（包含数值稳定性优化）
    :param x: 输入值（可以是标量、列表或 NumPy 数组）
    :return: 经过 Sigmoid 处理后的输出，范围在 (0, 1) 之间
    """
    return 1 / (1 + np.exp(-x))


    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        
        entropy = probs.entropy().sum(1)
        x_t = probs.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias           #真正取动作
        log_prob = probs.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action.detach(), log_prob,entropy,self.critic(x),
    def get_action_and_value_learn(self,x,action):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return action.detach(), probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
if __name__ == "__main__":
    # 1. 生成 1000 个测试数值（在 -10 到 10 之间均匀分布）
    num_points = 1000
    test_inputs = np.linspace(0, 10, num_points)
    
    # 2. 批量计算对应的 Sigmoid 输出
    outputs = sigmoid(test_inputs)
    
    # 3. 打印输入与输出结果（这里仅展示前 5 组作为示例，避免控制台刷屏）
    print("===== 测试数据预览 =====")
    for i in range(5):
        print(f"输入: {test_inputs[i]:.4f} | 输出: {outputs[i]:.6f}")
    print("... (共生成 1000 个测试点)")
    
    # 4. 绘制 Sigmoid 函数图像
    plt.figure(figsize=(10, 6))
    plt.plot(test_inputs, outputs, label='Sigmoid Function', color='#2E8B57', linewidth=2)
    
    # 添加辅助线和图表细节
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='y = 0.5')  # y=0.5 参考线
    plt.axvline(x=0, color='grey', linestyle='--', alpha=0.5)                 # x=0 参考线
    
    plt.title('Sigmoid Activation Function Visualization (1000 Points)', fontsize=14)
    plt.xlabel('Input (x)', fontsize=12)
    plt.ylabel('Output σ(x)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()