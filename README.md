# 这是AI生成的readme



# USV 智能导航算法实验仓库

本仓库用于验证无人船（USV）在仿真环境中的导航与避障策略，重点关注 **SAC** 和 **PPO** 两类强化学习方法的训练效果与奖励设计。项目中既包含训练逻辑，也包含与 ROS / Unity 仿真平台的交互代码。

---

## 2. 代码亮点

### 2.1 SAC 训练尝试

- 使用了 SAC 框架进行连续控制策略学习
- 观察到：**第一次训练**时模型通常能表现出一定的导航效果
- 但在后续训练中，可能出现模型输出趋向极大值的问题，推测与异常奖励信号或奖励尺度失控有关
- 后续实践表明SAC网络更新往往太过于激进不好调试

### 2.2 PPO 分阶段训练

- PPO 方案采用了分阶段训练思路
- 根据船体状态与任务进展，动态调整奖励和状态表达
- 代码中通过 `step_v1 / step_v2 / step_v3` 对训练阶段进行区分，便于分析不同阶段对策略学习的影响
- 另外，模型输入中加入了状态与动作的联合信息，提升了策略对当前行为的感知能力

### 2.3 奖励设计思路

项目中的奖励函数重点考虑了以下几个因素：

- 航向误差奖励（使船尽量朝向目标点）
- 目标距离奖励（当船靠近目标时提升奖励）
- 障碍距离惩罚（避免接近障碍物）
- 速度奖励（鼓励在安全范围内提高速度）
- 到达/碰撞终止奖励（用于强化任务完成与失败信号）

下面是当前实验中使用过的奖励设计核心思想：

```python
def setReward(self, state, action, heading, distance):
    obstacle_min_range = state[-2] / OBSTACLE_MIN_RANGE_W
    obstacle_r = (5 - obstacle_min_range) * (5 - obstacle_min_range) * 5

    distance_w = self.binary_cross_entropy_v1(
        1, (distance - self.arrive_distance) / 600
    ) / 3

    heading_r = distance_w * self.sigmoid_v1(5 - abs(heading / 9)) * 2
    if heading_r < -30:
        heading_r = -30

    speed_r = action[0, 0] * 7
    if distance <= 5:
        goal_r = (5 - distance) * (5 - distance) * 5
    else:
        goal_r = 0

    reward = heading_r + speed_r - obstacle_r + goal_r

    if self.arrive:
        reward += 100
    elif self.done:
        reward += -50

    return reward
```

其中：

- `distance_w` 会随着距离减小而增强对航向奖励的影响
- `sigmoid_v1` 用于把航向误差映射到一个相对平滑的奖励范围
- 该设计在实验中被认为对导航效果有一定帮助


###以上内容是第一版最终版参见PPO_NAV
---

## 3. 目录结构

```text
.
├── README.md                  # 项目说明文档
├── docs/                      # 竞赛文档与说明
├── model/                     # SAC已保存的模型文件
├── ppo_models/                # PPO 相关模型权重
├── traning_logs/              # PPO训练日志
└── NavAlg/
    └── NavAlg/
        ├── usvlib4ros/        # 核心 ROS/导航逻辑
        ├── requirements.txt   # 依赖列表
        └── setup.py           # 安装配置
```

---

## 4. 环境要求

建议环境：

- Python 3.9 / 3.10
- PyTorch
- NumPy
- OpenCV
- ROSLIBPY


## 5. 运行方式

### 5.1 运行主程序
SAC:
```bash
cd NavAlg/NavAlg
python usvlib4ros/main.py
```
PPO:
```bash
cd NavAlg/NavAlg
python usvlib4ros/main(1).py
```
### 5.2 运行 PPO 导航逻辑

项目中也提供了 PPO 导航控制实现，文件为：

- [NavAlg/NavAlg/usvlib4ros/user/PPO_NAV.py](NavAlg/NavAlg/usvlib4ros/user/PPO_NAV.py)

### 5.3 运行 SAC 导航逻辑

SAC 相关实现位于：

- [NavAlg/NavAlg/usvlib4ros/user/SAC_NAV.py](NavAlg/NavAlg/usvlib4ros/user/SAC_NAV.py)


---
