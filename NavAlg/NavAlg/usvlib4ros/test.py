import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体防止乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class LossEvaluator:
    def binary_cross_entropy_v1(self, y_true, y_pred):
        """计算二分类交叉熵损失的平方"""
        epsilon = 1e-15
        # 防止 log(0) 导致数值溢出
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss * loss  # 注意：代码实际是平方，并非注释中的三次方

# 1. 生成 1000 个测试点
np.random.seed(42)
n_points = 1000

# 模拟不同的预测概率 (0.001 到 0.999)
y_pred = np.linspace(0.001, 0.999, n_points)

# 为了展示效果，我们分别测试 y_true=1 和 y_true=0 的情况
# 这里以 y_true=1 为例：当 y_pred 越接近 1，损失越小；越接近 0，损失爆炸
y_true = np.ones(n_points) 

# 2. 计算损失
evaluator = LossEvaluator()
losses = []
for i in range(n_points):
    # 传入单点数组进行计算
    loss_val = evaluator.binary_cross_entropy_v1(np.array([y_true[i]]), np.array([y_pred[i]]))
    losses.append(loss_val)

losses = np.array(losses)

# 3. 绘制图像
plt.figure(figsize=(12, 6))
plt.plot(y_pred, losses, color='crimson', linewidth=2, label='y_true = 1')

# 添加一些视觉辅助
plt.title('二分类交叉熵损失平方 (1000个测试点)', fontsize=16)
plt.xlabel('预测概率 (y_pred)', fontsize=14)
plt.ylabel('Loss Squared', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# 因为靠近 0 时损失极大，使用对数坐标能更好展示趋势
plt.yscale('log')
plt.ylim(bottom=1e-6) 

plt.tight_layout()
plt.show()
print(evaluator.binary_cross_entropy_v1(1,0.9))