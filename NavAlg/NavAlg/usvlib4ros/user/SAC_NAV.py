"""
install pytorch in window10 cpu
python3.9 -m pip install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple
"""
import math
import time
import threading
import numpy as np
from usvlib4ros.navigation.usv_ros2_controller import Ros2Controller
from usvlib4ros.navigation.route_plan_service import RoutePlanService

from usvlib4ros.msg.global_data import GlobalData, DictToObject, Point, Constants
from usvlib4ros.msg.parameter import Parameter
from usvlib4ros.usvRosUtil import LogUtil
from usvlib4ros.user.SAC import SAC

"""笔记
1. 203行打印的heading完全不能用，它只会在某一轮开始时更新。但直接调用scandadata是可以的
2. 根据state更新逻辑，原代码heading和current_distence是完全不可信的数据，它们根本没有正确更新
3. heading已修复，具体是在step前，laser后赋值。current_distence只和reward有关，我就不管了，反正早晚要改掉reward
4. ======================================state[-4]被我*20来让神经网络赋予更多注意！因此heading值不要再用state获取
5.  state[-1]似乎不大对。不过我就根本没用这个值
6.不要轻易改state维度！！！！！！好不容易改好的！！！！！！
速度和旋转加入state
修改奖励
"""
import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Hyperparameters
N_ACTIONS = 2   #=========================================记得同步更改SAC_TEST的self.obs和宏
N_STATES = 94  # State dimension based on LiDAR data origin=184
MEMORY_CAPACITY = 2000
BATCH_SIZE = 128
LR_ACTOR = 0.0003
LR_CRITIC = 0.001
GAMMA = 0.99
K_EPOCHS = 80
EPS_CLIP = 0.2
ACTION_STD_INIT = 0.6
MAX_EPOCH = 4000

LASER_MAX_RANGE = 5.0        # 激光雷达有效最大距离(m)
LASER_FRONT_HALF_DEG = 180   # 前方扫描扇区角度(度)，仅用于碰撞检测
COLLISION_DISTANCE = 1.5     # 碰撞判定阈值(m)
ARRIVE_DISTANCE = 1.0        # 到达目标判定阈值(m)


class SAC_NAV:
    Instance = None

    """
    USV auto navigation service
    """

    def startService(self):
        self.navThread = threading.Thread(target=self.run)
        self.navThread.setDaemon(True)
        self.navThread.start()
        pass

    def __init__(self, ros_ctrl: Ros2Controller, global_data: GlobalData, xyzAxis=True):
        self.ros_ctrl: Ros2Controller = ros_ctrl
        self.global_data: GlobalData = global_data
        self.navThread = None

        #模型
        self.ppo_agent = SAC(ifload=False)
        self.next_state = None
        self.action_size = 5 #动作空间

        self.route = None   # 航线，
        self.destPoint = Point()  # 目标点  lng lat high speed
        self.destPointIndex = -1  # 目标点self.destPoint在导航点global_data.getInstance().route.points中的下标，-1表示无导航点。
        self.prevPoint = Point()
        self.prevPointIndex = -1
        self.max_distance = 0   # 船到目标的起始距离
        self.score = 0          # 评分
        self.episode_reward_sum = 0     # 本轮总分数
        self.arrive = False     # 到达
        self.done = False       # 碰撞障碍或越界
        self.arrive_distance = 1.0     # 最小距离，船到目标的距离低于此值时，认为到达目标。

        self.routePlaneService = RoutePlanService(wayPointRadius=self.arrive_distance, route=self.route)
        """启动时缓存下当前激光雷达数据，算法中检测激光雷达数据对象是否变化来判断是否收到新的数据"""
        self.last_laser_scan = global_data.laser_data
        pass

    def _extract_laser_features(self, scan) -> list:
        """
        从激光雷达数据提取特征，仅取前方180°扇区用于碰撞检测。

        假设ROS LaserScan的ranges按角度顺序排列，中间索引对应船体正前方。
        取中间180°范围内的数据点。
        """
        total_points = len(scan.ranges)
        # 前方180°占总扫描范围的一半，取数组中间部分
        half_count = total_points // 2
        start_idx = half_count // 2          # 前90°起始索引
        end_idx = start_idx + half_count     # 后90°结束索引

        scan_range = []
        for i in range(start_idx, end_idx):
            value = scan.ranges[i]
            if value == float('Inf') or value is None or np.isnan(value) or value > LASER_MAX_RANGE:
                scan_range.append(LASER_MAX_RANGE)
            else:
                scan_range.append(value)
        return scan_range


    def compute_bearing(self, lat1, lon1, lat2, lon2):
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lon = math.radians(lon2 - lon1)
        
        y = math.sin(delta_lon) * math.cos(lat2_rad)
        x = (math.cos(lat1_rad) * math.sin(lat2_rad) -
            math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon))
        bearing_rad = math.atan2(y, x)
        bearing_deg = (math.degrees(bearing_rad) + 360) % 360
        return bearing_deg
    
    
    def run(self):
        """"""
        """register parameter """
        """adjust parameter data update by topic thread"""
        self.registerParameter()
        global_step = 0
        while True:
            try:
                print("wait train button trigger ...")
                """self.ros_ctrl.device_data.task_status : 0 停止导航或结束训练;1 开启导航或训练 """
                while self.global_data.device_data.task_status == 0:
                    """unity端 训练模式，开始训练按钮没有按下，请按下开始训练按钮"""
                    time.sleep(1)
                    continue
                    pass

                for e in range(MAX_EPOCH):
                    if self.global_data.device_data.task_status == 0:
                        """unity端 训练模式，点击了停止训练按钮"""
                        print("Stop train ...")
                        break

                    """unity端 训练模式，第e轮训练"""
                    print(f"train {e} ...")
                    """初始化 unity端船位置和障碍物位置"""
                    print(f"Reset unity ...")
                    result = self.ros_ctrl.reset_unity()
                    time.sleep(1)  #等待1s ，等待self.global_data.device_data.reset_status 数据更新。
                    """等待unity复位完成, self.global_data.device_data.reset_status: 1 导航发出复位请求，2 unity回复复位完成"""
                    while self.global_data.device_data.reset_status != 2:
                        time.sleep(1)
                        continue

                    """设置为自动模式，启用算法输出控制船体（非自动模式，船体运动由其他模块输出控制）"""
                    self.ros_ctrl.set_auto_work()

                    """获取全局路线信息与导航算法路线信息比较"""

                    """重新加载航线"""
                    self.route = self.ros_ctrl.getRoute()
                    if len(self.route.points) == 0:
                        """没收到正确的航线数据"""
                        LogUtil.info("Error : len(route.points) is 0. ")
                        time.sleep(1)
                        continue
                    print(f"Route {self.route}...")

                    """开始新一轮的导航，清空缓存，加载航线"""
                    self.episode_reward_sum = 0
                    self.next_state = None
                    self.done = False
                    self.arrive = False
                    self.destPointIndex = -1
                    self.destPoint = None
                    self.max_distance = 0
                    self.reward = 0

                    self.global_data.route = self.route
                    self.__reloadNavigationRoute(self.route)

                    """第e次训练，最多3000步"""
                    startTime = time.time()
                    for t in range(3000):
                        if self.global_data.device_data.task_status == 0:
                            """unity端 训练模式，点击了停止训练按钮"""
                            print(f"Stop train step {t}...")
                            break
                        tempTime = time.time() - startTime

                        if tempTime > 300:
                            """本轮训练持续时间过长，主动结束本轮，开启下一轮"""
                            break

                        """第e轮训练的第t步"""
                        global_step += 1
                        self.navigationHandler(self.next_state,e,t,global_step)

                        

                        """monitor parameter data update by this thread"""
                        self.setMonitorParameterValue()

                        if self.done or self.arrive:
                            break

                        time.sleep(0.1)

                    
                pass
            except Exception as e:
                LogUtil.error(e)
                pass
            finally:
                time.sleep(0.02)#100ms



    def _is_last_waypoint_reached(self, current_distance: float) -> bool:
        """判断是否到达最后一个航路点。"""
        return (
            self.destPointIndex + 1 == len(self.route.points)
            and current_distance < self.arrive_distance
        )
        
        
    # getstate函数，获取机器人当前状态
    # 在功能测试钟，主要是获取到目标点之间的距离，只需要获取距离即可（使用gps）
    def getState(self, scan,heading,current_distance):
        """
        从激光雷达和传感器数据提取状态向量。

        Args:
            scan: 2D激光雷达数据对象(.ranges属性)
            heading: 船体航向(rad)
            current_distance: 到目标点的直线距离(m)

        Returns:
            状态向量 [laser_features..., heading, distance, obstacle_min_range, obstacle_angle]
        """
        scan_range = self._extract_laser_features(scan)
        obstacle_min_range = round(min(scan_range), 2)
        obstacle_angle = np.argmin(scan_range)
        LogUtil.debug(
            f"状态: heading={heading:.2f}, distance={current_distance:.2f}, "
            f"obstacle_min={obstacle_min_range:.2f}, obstacle_angle={obstacle_angle}"
        )
        # 碰撞检测
        if COLLISION_DISTANCE > obstacle_min_range > 0:
            LogUtil.info(f"检测到碰撞! obstacle_min_range={obstacle_min_range}")
            self.done = True

        # 到达检测
        if self._is_last_waypoint_reached(current_distance):
            self.arrive = True
        
        scan_range = self.normalize_feature(scan_range)  #正则化

        return np.append(scan_range , [heading*20, current_distance, obstacle_min_range, obstacle_angle])
    
    
    
    # step函数是用于执行一个动作并观察环境反馈的函数。它接收一个动作作为输入，并返回执行该动作后的新状态、奖励和完成标志。
    def step(self, state, action, laser_scan, heading, shipToNextWPDistance):
        """
        state: 状态空间
        action：随机动作
        laser_scan：2d激光数据
        heading：船体方向
        shipToNextWPDistance：船到目标点直线距离
        """
        state = state[0]


        state = self.getState(laser_scan, heading, shipToNextWPDistance)

        reward = self.setReward(state, action, self.max_distance)

        return state, reward,  heading

    def registerParameter(self):
        """"""
        """重启导航模块后 清空已有参数列表"""
        self.ros_ctrl.initParameterList()
        """adjust parameter data update by topic thread"""

        """导航点有效半径，船位于此半径内即为达到导航点"""
        name = '/usv/auto/plan/attainRadius'
        attainRadius = Parameter(name=name, dataType='float', defaultData=2)
        result = self.ros_ctrl.registParameter(attainRadius)
        if result:
            self.global_data.parameterAdjustMap.update({name: attainRadius})
            self.routePlaneService.wayPointRadius = attainRadius.value

        param2 = Parameter(name='/usv/auto/param2', dataType='int', defaultData=[20])
        result = self.ros_ctrl.registParameter(param2)
        if result:
            self.global_data.parameterAdjustMap.update({'/usv/auto/param2': param2})

        param3 = Parameter(name='/usv/auto/param3', dataType='int', defaultData=[30, 31])
        result = self.ros_ctrl.registParameter(param3)
        if result:
            self.global_data.parameterAdjustMap.update({'/usv/auto/param3': param3})

        param4 = Parameter(name='/usv/auto/param4', dataType='float', defaultData=[40.1, 40.2])
        result = self.ros_ctrl.registParameter(param4)
        if result:
            self.global_data.parameterAdjustMap.update({'/usv/auto/param4': param4})

        param6 = Parameter(name='/usv/auto/param6', dataType='str', defaultData='some msg')
        result = self.ros_ctrl.registParameter(param6)
        if result:
            self.global_data.parameterAdjustMap.update({'/usv/auto/param6': param6})

        """monitor parameter """
        # TODO

        pass

    def setMonitorParameterValue(self):
        """monitor parameter """

        pass

    def __loadVehiclePoseInfo(self):
        """
        isAuto  : True 自动导航;False 非自动导航
        isReturn:True 自动返航;False 非自动返航
        lng     : 航行器当前经度
        lat     : 航行器当前纬度
        realSpeed:航行器当前速度 m/s
        realRotateSpeed:航行器当前角速度 °/s
        """
        heading = 0
        workModel = self.global_data.device_data.work_model
        isReturn = workModel == Constants.WorkMode.AutoReturn
        isAuto = workModel == Constants.WorkMode.Auto

        # lng = global_data.getInstance().gpsInfo.lng
        # lat = global_data.getInstance().gpsInfo.lat
        # heading = global_data.getInstance().gpsInfo.heading
        # realSpeed = global_data.getInstance().gpsInfo.speed
        # realRotateSpeed = 0.0 # TODO °/s

        # if global_data.getInstance().autoCommand.poseModel == Constants.PoseMode.UintyPose:
        pose = self.global_data.scada_data.pose
        lng = pose.lng
        lat = pose.lat
        if self.destPoint is not None:
            heading = self.compute_bearing(lat,lng,self.destPoint.lat, self.destPoint.lng)
        if heading > 180:
            heading = heading - 360
        realSpeed = pose.speed
        realRotateSpeed = pose.rotate_speed

        return isAuto, isReturn, lng, lat, heading, realSpeed, realRotateSpeed

    def __reloadNavigationRoute(self, route):
        """return True:new route"""
        """更新route"""
        self.route = route  # load route info from global data
        self.routePlaneService.reset(route=self.route)
        return True

    def get_laser_scan(self,timeout):
        laser_start_time = time.time()
        """
        self.global_data.laser_data : ros2 提供的最新的激光雷达数据
        self.last_laser_scan ： 算法上次处理的激光雷达数据
        是同一个对象则未收到新的数据，超时返回None
        """
        while self.global_data.laser_data == self.last_laser_scan:
            time.sleep(0.1)
            get_laser_use_time = time.time() - laser_start_time
            if get_laser_use_time >timeout:
                return None
            pass

        laser_scan = self.global_data.laser_data
        return laser_scan

    
    def navigationHandler(self,state,episode,step,global_step):
        """
        导航算法:
        """
        try:

            """
            获取航行器姿态和位置
            isAuto  : True 自动导航;False 非自动导航
            isReturn:True 自动返航;False 非自动返航
            lng     : 航行器当前经度
            lat     : 航行器当前纬度
            realSpeed:航行器当前速度 m/s
            realRotateSpeed:航行器当前角速度 °/s
            """
            isAuto, isReturn, lng, lat, self.routePlaneService.degreeAShip, realSpeed, realRotateSpeed = self.__loadVehiclePoseInfo()

            """路径规划：航线中多个点进行有序导航"""
            self.routePlaneService.setCurrentPos(lng, lat, isReturn)

            """获取路径规划结果"""
            try:
                nextPointIndex = self.routePlaneService.curNextIndex
                prevPointIndex = self.routePlaneService.curPrevIndex

                nextPoint = self.route.points[nextPointIndex]
                if prevPointIndex < 0 or prevPointIndex >= len(self.route.points):
                    prevPointIndex = nextPointIndex

                if nextPointIndex != self.destPointIndex or nextPoint != self.destPoint:
                    """切换目标点,记录此时到目标点的距离"""
                    self.max_distance = self.routePlaneService.distanceMetersShip2NextWP
                    LogUtil.info(f"Goto point[{nextPointIndex}] = {nextPoint}, distance = {self.max_distance}")

                """当前导航目标点"""
                self.destPointIndex = nextPointIndex
                self.destPoint = self.route.points[self.destPointIndex]  # 目标点
                self.prevPointIndex = prevPointIndex
                self.prevPoint = self.route.points[self.prevPointIndex]  # 目标点
            except Exception as e:
                LogUtil.error(e)

            if self.destPointIndex == -1:
                """无导航目标点"""
                return

            """自动导航相关数据"""
            destLng = self.destPoint.lng
            destLat = self.destPoint.lat
            prevLng = self.prevPoint.lng
            prevLat = self.prevPoint.lat
            shipToNextWPDistance = self.routePlaneService.distanceMetersShip2NextWP
            heading = self.routePlaneService.degreeAShip#=========================新增。原逻辑没有更新heading
            print("==========heading",heading)
            """等待新的激光雷达数据"""
            laser_scan = self.get_laser_scan(timeout=2)     # 2s 超时
            if laser_scan is None:
                print("Get 2d laser scan timeout.")
                return False
            self.last_laser_scan = laser_scan

            """*******************自定义导航算法 Start***************************"""
            """第一次进入导航"""
            if state is None:
                state = self.getState(laser_scan, heading, shipToNextWPDistance)
            state = np.expand_dims(state, axis=0)     #增添维度适应SAC类要求
            action,time = self.ppo_agent.run(state,self.reward,self.done or self.arrive,global_step)
            #=============================在SAC里做正则，因为这里的state的后继维度还有用
            print("===========time",time)
            
            adviseSpeed = action[0,0]*100
            adviseRotate = action[0,1]*100
            self.next_state, self.reward, advisedHeading, =\
                self.step(state.tolist(),action,laser_scan,heading, shipToNextWPDistance)
            self.episode_reward_sum += self.reward

            """算法输出结果保存到global_data"""
            self.global_data.updateAlgorithmOutput(episode, step, int(self.episode_reward_sum), self.reward, MAX_EPOCH, 2)
            self.global_data.updateThrottleRudderOutput(adviseSpeed, adviseRotate, advisedHeading, nextPointIndex,
                                                        shipToNextWPDistance)
            print(f"Step: {step}, Action: {action}, Reward: {self.reward}, Done: {self.done}")

            if self.arrive or self.done:
                print(f"Episode ended at step {step}, total reward: {self.episode_reward_sum}")
                return True
            """---------------------------自定义导航算法 END-------------------------"""
            """get result"""
            LogUtil.info("Goto index = %s, heading : %s = > %s .distance=%s .speed = %s , rotate = %s ." % (
                self.destPointIndex, heading, advisedHeading, shipToNextWPDistance, adviseSpeed, adviseRotate))
            pass
        except Exception as e:
            LogUtil.error(e)
        return False

    def setReward(self, state, action, max_distance):       # 少一个方向reward    scanreward和obreward只能有一个好像
        obstacle_min_range = state[-2]
        current_distance = state[-3]
        scan_reward = 0
        ob_reward = 0

        if current_distance > 1:
            scan_reward = 1 - (current_distance / max_distance)
            if scan_reward < 0:
                scan_reward = scan_reward * 2
            else:
                scan_reward = scan_reward * 5

        if current_distance > max_distance:
            self.max_distance = current_distance

        

        distance_rate = 2 ** (current_distance / self.max_distance)    # self.dis是全局的距离吗 这里的reward值都是正数
        #forward_reward = ((round(yaw_reward[action] * 5, 2)) * distance_rate)

        if obstacle_min_range < 3:
            ob_reward = -5
        if current_distance < 3:
            ob_reward = 1
        # else:
        #     ob_reward = 1                                       #考虑这里就不要这个了

        # reward = scan_reward * 0.5  + forward_reward * 0.3 + ob_reward * 0.2        #尝试使用加权奖励
        reward = scan_reward * 0.6 + ob_reward * 0.2 #  + forward_reward * 0.2

        if self.arrive:
            LogUtil.info("Goal!!")
            reward += 1000
        elif self.done:
            LogUtil.info("Collision!!")
            reward += -500
        return reward
    
    def normalize_feature(self,feat: np.ndarray):
        """逐样本 0均值1方差规范化 (Z-Score)"""
        # 计算均值和标准差，利用 keepdims=True 保持维度以便广播相减/除
        mean = np.mean(feat, axis=-1, keepdims=True)
        std = np.std(feat, axis=-1, keepdims=True) + 1e-6  # 防除零
        
        feat_norm = (feat - mean) / std
        return feat_norm