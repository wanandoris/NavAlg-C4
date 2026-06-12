经过测试目前阶段第一次训练可以达到训练效果但第二次时模型就直接训成模型输出的最大值了推测是有极端reward混进去导致的
model下1024的pt模型是有一定训练效果的，但2024则训崩了，只会输出能输出的最大值
此代码的亮点是
1. 用了SAC但由于没有截断函数之类的效果可能不如PPO
2. 设置奖励函数代码如下  经测试的确是有效果的.其中 distance_w 是 heading_r计算时的参数，与目标距离越小，对于heading奖励的放大效应越大    heading奖励则是在很大的值下趋于平滑
       def setReward(self, state,action,heading,distance):       # 少一个方向reward    scanreward和obreward只能有一个好像
        obstacle_min_range = state[-2] / OBSTACLE_MIN_RANGE_W         #====我觉得可以不加，因为撞击后扣得已经够模型受得了
        obstacle_r = (5-obstacle_min_range)*(5-obstacle_min_range)*5
        print("=======obstacle_r",obstacle_r,"======dis_to_obstacle===",obstacle_min_range)
        distance_w = self.binary_cross_entropy_v1(1,(distance-self.arrive_distance)/600)/3     # ==========或许公式要改  与目标点最小距离，达到即判定抵达，与目前距离的交叉熵的平方除以三。我觉得这个数值合适
        print("=====distance=====",distance-self.arrive_distance,"=======dis_w=======",distance_w)
        
        heading_r = distance_w*self.sigmoid_v1(5-abs(heading/9))*2  #===-1~1，对准时在0.9左右        #=============或许要改
        if heading_r < -30:
            heading_r = -30          #防止在目标旁边转向太猛扣太多，从而不敢去目标
        print("=====heading=====",heading,"======heading_r",heading_r)

        speed_r = action[0,0]*7                            #==========速度越大奖励越高。具体还得测试
        if distance<=5:
            goal_r = (5-distance)*(5-distance)*5
        else:
            goal_r = 0
        reward = heading_r+speed_r-obstacle_r+goal_r     #朝向+速度（较小）-障碍距离+目标距离（与障碍抵消）
        print("=======total_r",reward)
        if self.arrive:
            LogUtil.info("Goal!!")
            reward += 100
        elif self.done:
            LogUtil.info("Collision!!")
            reward += -50
        return reward
    
    def binary_cross_entropy_v1(self,y_true, y_pred):
        """计算二分类交叉熵损失的二次方"""
        epsilon = 1e-15
        # 防止 log(0) 导致数值溢出
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss*loss
    def sigmoid_v1(self,x):
        """
        对于结果进行处理，最后为-1~1
        Sigmoid 激活函数（包含数值稳定性优化）
        :param x: 输入值（可以是标量、列表或 NumPy 数组）
        :return: 经过 Sigmoid 处理后的输出，范围在 (0, 1) 之间
        """
        result = 1 / (1 + np.exp(-x))
        return (result-0.5)*2
