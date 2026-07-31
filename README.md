# C4 流程说明
这份**README**重点是介绍目前整个仿真工作流程：

## 1. 当前流程总览
### 1.1 智能体
目前的智能体有一个**<font style="background-color:#FBDE28;">96</font>**维的离散状态空间，由：

**90长的雷达数组 + 速度 + 角速度 + 与目标角度差 + 与目标点距离 + 与最近障碍物的距离 + 与最近障碍物的夹角  
**有一个2维的连续动作空间，由：  
第一维：**预期转角**  
第二维：**预期速度**

****

这两个输出值里，预期转角会被`**PID**`映射为应转的角速度量，这个量可能很大。  
而预期速度会被`ACTION_TO_SPEED_CONTINOUS_SCAL`，缩放到正常比例，但是这个速度值没那么重要，可以先不管。

### 1.2 训练流程
目前的训练流程，采取`apf-pid`和`ppo`切换的形式。  
`apf_pred`是基于**人工势场法**计算出的预期角度方向，其兼顾了目标的吸引作用与障碍物的排斥作用，具体实现方式可见`docs/PPO_APF.md`，在其提供的帮助下，可以使得船几乎正确的移动（可以通过在`nav_2.cpp`中将`**ENABLE_APF_DEBUG_VIEW**`设为`True`以验证准确性），而每一个`step`中，`ppo`的策略会同样选出一个角度作为`ppo_pred`供船转向。

在初期，`ppo_pred`的探索性会使得成功率过低，而我们借助一个ε因子来丰富样本池，其计算方式如下:

$ \epsilon =max(\epsilon_{min}, e^{-k·max(0,episode)}) $

也就是说，在初期，`**PID**`会近乎百分百的采取`apf_pred`，使得初期`**PPO**`可以获得足够多的成功样本而不是盲目撞墙，随后$ \epsilon $会随时间衰减到$ \epsilon_{min} $,届时策略将完全由`PPO`主导。

<img src="https://cdn.nlark.com/yuque/0/2026/png/60743459/1781714608073-79bd20a6-0a89-4ec6-8ac4-2ef637f78ca7.png?x-oss-process=image%2Fcrop%2Cx_0%2Cy_0%2Cw_982%2Ch_297" width="524" title="apf-pid主导下的初期成功率" crop="0,0,0.9704,1" id="ucec51c2c" class="ne-image">



既然你看到了这里，那么允许你去`nav_2.cpp`里把`ROTATE_CONTROL_MODE`改成`PID`，跑跑仿真爽一把😋

### 1.3 奖励设置
1. 当前总 reward 主要由以下几部分组成：
2. `distance_reward`
3. `heading_reward`
4. `obstacle_reward`
5. `time_reward`
6. `step_penalty`
7. `arrive_bonus / collision_penalty`

#### 1.3.1 distance_reward
当前的 `distance_reward` 由两部分组合出来的：

+ `progress_reward`
+ `proximity_reward`

其中：

+ `progress_reward` 奖励“相较于上一步，当前是否更接近目标”,其作为一个进度项，可视为：

$ progress = prev\ distance - current\ distance $

+ `proximity_reward` 奖励“当前本身是否已经比较接近目标”,其作为一个非线性的接近项，当前采用类似吸引势场的指数形式，距离越近值越大(但不会大过`arrive_bonus`，权重也不高，防止刷分)

$ proximity= exp\{-k\frac{current \ distance}{distance \ scale}\} $

`distance scale`一般取本轮中距离目标点的最远距离。

#### 1.3.2 heading_reward
`heading_reward`计算的是当前航向距上文介绍的`apf_pred`的插值，`apf_pred`的计算过程可以看作**目标引力**和**障碍物斥力**两力进行合成后的结果，并且会在引力和斥力前分别乘上`apf_attractive_gain`  
   ` apf_repulsive_gain`作大小调整

#### 1.3.3 obstacle_reward
`obstacle_reward` 本质上是一个斥力场惩罚项。其计算方式同样与`docs/PPO_APF.md`相同。

#### 1.3.4 time_reward
$ w(t) = w_{\min} + (1 - w_{\min}) \exp\left(-4 \cdot \operatorname{clip}\left(\frac{t}{T_{\max}}, 0, 1\right)\right) $

随着时间增大，该惩罚会指数级增大。

#### 1.3.5 其它
还有几个非常直接的终止项：

+ `step_penalty`  
每一步都会额外扣一点，防止智能体无意义拖时间
+ `reward_arrive_bonus`  
成功到达目标时给一个明确的大正奖励，但是原先的1000会使得我们设置的奖励都太稀疏了，我改成80
+ `reward_collision_penalty`  
碰撞时给一个明确的负奖励，同样的问题，我改成-50



## 2. 训练结果查看
每次训练都会新建一个结果目录，例如：

+ `Results/ppo_nav_20260617_230722/`

这个目录保存的是一次完整实验的记录。

### 2.1 `episode_metrics.csv`
这个文件按 episode 记录，也就是每一轮训练一行。

比较重要的列包括：

+ `episode_return`  
这一轮的总 reward，是最直观的单轮表现指标
+ `steps`  
这一轮总共跑了多少步
+ `episode_time_sec`  
这一轮总耗时
+ `arrived`  
是否成功到达目标
+ `collided`  
是否碰撞
+ `timeout`  
是否因为超时结束
+ `success_rate_total`  
截止当前 episode 的累计成功率
+ `collision_rate_total`  
截止当前 episode 的累计碰撞率
+ `avg_episode_time_total`  
截止当前的平均单轮耗时
+ `avg_steps_total`  
截止当前的平均步数

### 2.2 `summary_metrics.csv`
这个文件按 PPO update 记录，也就是每次策略更新为一行。

比较重要的列包括：

+ `update_step`  
当前是第几次 PPO 更新
+ `buffer_size`  
这次更新用了多少样本
+ `actor_loss`  
当前 actor 的 loss
+ `critic_loss`  
当前 critic 的 loss
+ `total_loss`  
当前总 loss
+ `entropy`  
当前策略熵
+ `total_episodes`  
到这次更新为止，总共已经跑了多少轮
+ `success_rate_total`  
累计成功率
+ `collision_rate_total`  
累计碰撞率
+ `avg_episode_time_total`  
累计平均单轮耗时
+ `avg_steps_total`  
累计平均步数
+ `auc_return`  
累计回报面积，可以粗略理解为训练期间总体收益的累计表现
+ `first_arrive_episode`  
第一次成功到达目标出现在第几轮

### 2.3 `ppo_update_metrics.csv`
这个文件聚焦 PPO 本身的更新情况。

主要看这些列：

+ `actor_loss`
+ `critic_loss`
+ `total_loss`
+ `entropy`
+ `buffer_size`

### 2.4 `checkpoints/`
这里保存的是周期导出的模型权重，例如：

+ `PPO_ship_obstacle_0.pth`
+ `PPO_ship_obstacle_100.pth`

它们的主要用途是：

+ 保存实验过程中的中间策略
+ 回头比较不同阶段的模型效果
+ 后续单独加载某个 checkpoint 做验证

### 2.5 TensorBoard 日志
除了 `Results/` 中的 csv 和图片外，TensorBoard 日志也会同步写到：

+ `runs/ppo_nav_时间戳/`

如果要开 TensorBoard，可以直接运行：

```powershell
tensorboard --logdir d:\C4\runs
```

TensorBoard 更适合实时观察训练过程，尤其适合边训练边看：

+ reward 曲线
+ success / collision 曲线
+ actor / critic loss
+ entropy
+ heading 相关指标

## 3. 调参
下面只提供需要/有必要调的参数

### 3.1 PPO相关
```python
N_STATES = 96          # 状态维度: 前方180°激光点数(约90) + speed + rotated_speed + angel_diff + distance + obstacle_min_range + obstacle_angle
LR_ACTOR = 0.00003     # PPO中神经网络学习率，越大越容易过拟合，PPO原论文中ACTOR和CRITIC均为3*10-4j
LR_CRITIC = 0.0001
GAMMA = 0.99           # 折扣因子
K_EPOCHS = 8           # PPO更新轮数(我测这个还算不错)
ACTION_STD_INIT = 0.5  # 连续动作标准差初始化(越大初始探索性越强)
UPDATE_INTERVAL = 256  # PPO更新间隔(步数)
MIN_BUFFER_SIZE_FOR_UPDATE = 256
MIXED_ROTATE_CONTROL = True  	# 是否启用APF-PID与PPO混合切换
TEACHER_EPSILON_DECAY = 0.008   # APF-PID作为专家的概率衰减系数
TEACHER_EPSILON_MIN = 0.20      # 专家概率下限
```

我还没有尝试过的:

+ 90维的雷达数组可能占据过多状态空间大小，你可以试着在`_extract_laser_features`函数中尝试将

```python
for i in range(0, min(len(reordered_ranges), 180), 2):
```

替换为4个一取，使雷达数据只有45维，我感觉在可视化中45维还算密集

### 3.2 Reward 相关
```python
REWARD_ARRIVE_BONUS = 80	   # 到达奖励
REWARD_COLLISION_PENALTY = -50 # 碰撞惩罚
REWARD_WEIGHT_DISTANCE = 2.4   # 距离权重
REWARD_WEIGHT_OBSTACLE = 0.9   # 障碍物权重
REWARD_WEIGHT_HEADING = 1.0    # 朝向权重
REWARD_WEIGHT_TIME = 0.2       # 时间权重
REWARD_PROGRESS_WEIGHT = 0.7   # 进步权重（在distance中）
REWARD_PROXIMITY_WEIGHT = 0.3  # 接近权重（在distance中）
REWARD_PROXIMITY_EXPONENT = 2.4 			# 接近权重指数k值（e^kx）			
REWARD_STEP_PENALTY = -0.005				# 步数小惩罚
REWARD_MIN_ARRIVE_TIME_WEIGHT = 0.35		# 控制arrive_bonus大小，到达的越早拿的越多，越晚不会低于REWARD_MIN_ARRIVE_TIME_WEIGH
REWARD_APF_ATTRACTIVE_GAIN = 1.0			# 引力缩放系数
REWARD_APF_REPULSIVE_GAIN = 8.0				# 斥力缩放系数
REWARD_APF_OBSTACLE_INFLUENCE_RANGE = 2.5	# APF开始考虑障碍物斥力的距离
REWARD_TIME_EXPONENT = 1.4					#  时间权重指数k值（e^kx）
```

如果有好的想法，你也可以按照上面奖励里公式的介绍去改公式

### 3.3 导航/PID相关
```python
LASER_MAX_RANGE = 8.0        # 激光雷达有效最大距离(m)
COLLISION_DISTANCE = 0.6     # 碰撞判定阈值(m)
ARRIVE_DISTANCE = 1.5        # 到达目标判定阈值(m)
DEFAULT_SPEED = 1.0          # 默认速度(m/s)
OBSTACLE_SLOW_RANGE = 4.0    # 进入此范围开始减速(m)
TARGET_SLOW_RANGE = 3.0      # 接近目标时减速阈值(m)
ANGULAR_VELOCITY_MAX = 100   # 策略/奖励/预测使用的内部最大角速度(°/s)
PPO_TARGET_HEADING_MAX_OFFSET = 90.0  # PPO目标航向最大偏转角(度)
PPO_HEADING_SLEW_LIMIT_ENABLED = True # 是否限制相邻PPO目标航向的突变
PPO_HEADING_MIN_CHANGE = 70.0         # PPO相邻输出允许的最低变化范围(度)
PPO_HEADING_ERROR_MARGIN = 20.0       # 在上一步PPO/APF角差上增加的余量(度)
ACTION_TO_SPEED_CONTINOUS_SCALE = 120 # 连续动作映射到速度缩放因子
CONTROL_DT = 0.01            # 控制周期(s),用于连续动作
ROTATE_CONTROL_MODE = "PPO"  # 可选: "PPO" | "PID"
HEADING_PID_KP = 1.2
HEADING_PID_KI = 0.02
HEADING_PID_KD = 0.15
HEADING_PID_INTEGRAL_LIMIT = 60.0
```



### 3.4 其它
```python
# ==================== 其它 ====================
MAX_EPOCH = 4000       # 最大训练轮数
MAX_STEP_PER_EPISODE = 500   # 每轮最大步数
MAX_EPISODE_TIME = 300  # 每轮最大时间(秒)
CHECKPOINT_INTERVAL = 100  # 模型保存间隔(轮数)
ENABLE_APF_DEBUG_VIEW = False         # 是否打开APF方向实时调试窗口
APF_DEBUG_WINDOW_NAME = "APF Heading Debug"
```

### 3.5 PPO 结构改动
如果你想改的是算法本身就去看PPO  
我常改的有：`std_log`的clamp，`ActorCritic`网络结构（其实只是加过一次`nn.LayerNorm(hidden_dim)`层，效果要是不好可以试着回滚版本补回原来官方示例的那样)

### 3.6 时间刻
现在还有一个比较棘手的地方，就是按我现在的更新策略（既要有效batch够大，每个episode质量不能太低），大概跑了380个回合，成功`update`150次也并无效果。我问孟哥如果希望算法见效，估计要跑几千次`update`

但是我和我朋友都感觉就这个破环境部署到服务器租显卡有点难度。如果大家希望见效快一点，可以选择开小图加快仿真时间刻，具体需改动：

```python
# ==================== 时间刻 ====================
TASK_WAIT_SLEEP = 0.01          # 等待训练触发轮询间隔(秒)
EMPTY_ROUTE_SLEEP = 0.01        # 航线为空时的等待间隔(秒)
STEP_SLEEP = 0.01               # 每步主循环结束等待间隔(秒)
FINAL_SLEEP = 0.2               # 异常/结束后的等待间隔(秒)
RESET_START_SLEEP = 0.01        # reset_unity后首次等待(秒)
RESET_STATUS_SLEEP = 0.01       # 等待reset_status轮询间隔(秒)
LASER_TIMEOUT = 0.2             # 等待激光数据超时(秒)
LASER_POLL_SLEEP = 0.01         # 激光轮询间隔(秒)
RESET_SETTLE_DELAY = 0.2  		# 每轮复位后等待仿真刷新(秒)

MAX_EPISODE_TIME = 300  		# 每轮最大时间(秒)（同样与时间刻相关）
CONTROL_DT = 0.01            	# 控制周期(s),用于连续动作 （同样与时间刻相关）
```

### btw
每次实验建议记改了哪些参数、最终看的是哪几个指标

最简单可以补一个文本，比如：

```latex
2026-06-18
baseline: Results/ppo_nav_20260617_230722
change: reward_weight_obstacle 0.9 -> 1.2
change: apf_repulsive_gain 8.0 -> 10.0
observe: success_rate_total, collision_rate_total, episode_return
```

这样后面对比会轻松很多。



## 4.我目前做了什么
1.改状态空间，连续二维动作空间和动作空间形式，actor，critic网络结构，使动作方差可学习，添加GAE奖励。  
2.将输出动作映射为可执行动作，通过介入pid控制器

3.设置apf-heading奖励，非线性目标进步奖励和障碍物惩罚  
4.设置apf-pid 和 ppo 策略切换

5.雷达数组修复，目标点错误下发修复，目标点错识别修复

6.logger和可视化，我建议量化标准按csv里我用的来





我觉得我能做的和能想到的都做了，如果有想到的可行的，算法或代码方面可以对照着进行补充。如果没有的话就是需要调可行的参数了。有能力的话录点可行的视频（我建议对于同组实验**每隔50-100轮录一次，不同组实验对比相同轮数效果**，也可以录可视化图）或者你们谁写个脚本把**CSV导成曲线图**也可以，后面交报告和ppt用。据孟哥说出效果本地得跑上个几千次，我建议是加快时间刻先跑着，剩下的调参的效果大家加油。



---

感谢我的队友们与挚友qhy :)
