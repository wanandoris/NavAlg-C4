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
from usvlib4ros.user.PPO import PPO

"""笔记
1. 203行打印的heading完全不能用，它只会在某一轮开始时更新。但直接调用scandadata是可以的
2. 根据state更新逻辑，原代码heading和current_distence是完全不可信的数据，它们根本没有正确更新
3. heading已修复，具体是在step前，laser后赋值。current_distence只和reward有关，我就不管了，反正早晚要改掉reward
4. ======================================state[-4，-3，-2，-1]被我*20来让神经网络赋予更多注意！因此heading值不要再用state获取
5.  state[-1]似乎不大对。不过我就根本没用这个值
6.不要轻易改state维度！！！！！！好不容易改好的！！！！！！

7.==============================待办：测试障碍物信息 self.route.obstacles[]     无数据！！！！！！！！！！
8.待测：船的极限速度

速度和旋转加入state
修改奖励
"""
import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Hyperparameters

MAX_EPOCH = 4000

LASER_MAX_RANGE = 5.0        # 激光雷达有效最大距离(m)
COLLISION_DISTANCE = 2     # 碰撞判定阈值(m)

LOAD_MODEL_STEP = 10         #要加载的模型名称最后step的数字

OBSTACLE_MIN_RANGE_W = 5    #这个是obstacle_min_range乘以多少放到state里。写在这是因为算奖励时从state里获取到障碍距离
class PPO_NAV:
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
        self.ppo_agent = PPO(ifload=False,file_path = "D:\\大赛资源\\智能导航C4-2026\\unpack\\NavAlg-C4-v1\\ppo_models\\.pt")
        self.next_state = None
        self.action_size = 5 #动作空间
        self.isbug = 0
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
        #=============碰撞判定超参数在上边
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
                if value > 10 :
                    value = 10
                scan_range.append(value)
        return scan_range
    
    def _extract_laser_features_v1(self, scan) -> list:
        
        total_points = len(scan.ranges)
        

        scan_range = []
        for i in range(total_points):
            value = scan.ranges[i]
            if value == float('Inf') or value is None or np.isnan(value) or value > LASER_MAX_RANGE:
                scan_range.append(LASER_MAX_RANGE)
            else:
                if value > 10 :
                    value = 10
                scan_range.append(value)
        return scan_range


    @staticmethod
    def _normalize_signed_angle_diff(angle: float) -> float:
        return (angle + 180) % 360 - 180
    @staticmethod
    def _normalize_heading_360(angle: float) -> float:
        return angle % 360
    def _get_current_heading(self) -> float:
        """获取当前航向。"""
        pose = self.global_data.scada_data.pose
        heading = pose.yaw
        return heading - 360 if heading > 180 else heading
    def compute_bearing(self, lat1, lon1, lat2, lon2):
        
        """
        返回角度误差heading
        """
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lon = math.radians(lon2 - lon1)
        
        y = math.sin(delta_lon) * math.cos(lat2_rad)
        x = (math.cos(lat1_rad) * math.sin(lat2_rad) -
            math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon))
        bearing_rad = math.atan2(y, x)
        bearing_deg = (math.degrees(bearing_rad) + 360) % 360
        if bearing_deg>180:
            bearing_deg = bearing_deg - 360
            
        heading = self._get_current_heading()
        
        bearing_deg = self._normalize_signed_angle_diff(bearing_deg - heading)
        
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
                    self.isbug = 0
                    for t in range(150):
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
                        starttime = time.time()
                        
                        #self._check_route_update()   #=========================新增，测试是否获取障碍点

                        self.navigationHandler(self.next_state,e,t,global_step,e)

                        

                        """monitor parameter data update by this thread"""
                        self.setMonitorParameterValue()
                        endtime = time.time()
                        print("=========time",endtime-starttime)
                        if self.done or self.arrive:
                            break
                        if self.isbug == 15:
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
        scan_range = self._extract_laser_features_v1(scan)
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
        
        #scan_range = self.normalize_feature(scan_range)  #正则化
        return np.append(scan_range , [heading*20, current_distance*10, obstacle_min_range*OBSTACLE_MIN_RANGE_W, obstacle_angle*20])   #乘上一个数是为了让模型放更大的注意在这些参数上
    
    
    
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

        print("===========heading",heading,"========shipToNextWPDistance",shipToNextWPDistance)
        state = self.getState(laser_scan, heading, shipToNextWPDistance)

        reward = self.setReward(state,action,heading,shipToNextWPDistance)

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
    
    def _load_vehicle_pose_info(self) -> tuple:
        """获取载体位置和姿态信息。"""
        workModel = self.global_data.device_data.work_model
        isReturn = workModel == Constants.WorkMode.AutoReturn
        isAuto = workModel == Constants.WorkMode.Auto

        pose = self.global_data.scada_data.pose
        lng, lat = pose.lng, pose.lat
        heading = pose.yaw
        if heading > 180:
            heading -= 360

        return isAuto, isReturn, lng, lat, heading, pose.speed, pose.rotate_speed

    def _calc_distance_to_target(self, lng1: float, lat1: float, lng2: float, lat2: float) -> float:
        rad_lat1 = math.radians(lat1)
        rad_lat2 = math.radians(lat2)
        delta_lat = rad_lat1 - rad_lat2
        delta_lng = math.radians(lng1) - math.radians(lng2)
        distance = 2 * math.asin(
            math.sqrt(
                math.sin(delta_lat / 2) ** 2
                + math.cos(rad_lat1) * math.cos(rad_lat2) * math.sin(delta_lng / 2) ** 2
            )
        )
        return round(distance * 6378.137 * 1000, 1)
    
    def _update_navigation_target(self, pose_info: tuple):
        """更新路径规划目标点,返回导航上下文。"""
        isAuto, isReturn, lng, lat, _, _, _ = pose_info
        self.routePlaneService.setCurrentPos(lng, lat, isReturn)

        try:
            nextPointIndex = self.routePlaneService.curNextIndex
            prevPointIndex = self.routePlaneService.curPrevIndex
            nextPoint = self.route.points[nextPointIndex]

            if prevPointIndex < 0 or prevPointIndex >= len(self.route.points):
                prevPointIndex = nextPointIndex

            if nextPointIndex != self.destPointIndex or nextPoint != self.destPoint:
                self.max_distance = 0.0
                LogUtil.info(f"切换至航点[{nextPointIndex}]={nextPoint}, 距离={self.max_distance:.1f}m")

            self.destPointIndex = nextPointIndex
            self.destPoint = self.route.points[self.destPointIndex]
            self.prevPointIndex = prevPointIndex
            self.prevPoint = self.route.points[prevPointIndex]

            #现在的位置到目标点的距离和角度都是手算
            shipToNextWPDistance = self._calc_distance_to_target(lng, lat, self.destPoint.lng, self.destPoint.lat)
            degreeAship = self.compute_bearing(lat, lng, self.destPoint.lat, self.destPoint.lng)
            if self.max_distance <= 0:
                self.max_distance = shipToNextWPDistance

        except Exception as e:
            LogUtil.error(f"路径规划异常: {e}")

        if self.destPointIndex == -1:
            return None

        return {
            'nextPointIndex': nextPointIndex,
            'shipToNextWPDistance': shipToNextWPDistance,
            'degreeAship' : degreeAship
        }
    
    def navigationHandler(self,state,episode,step,global_step,e):
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
            

            """获取路径规划结果"""
            
            pose_info = self._load_vehicle_pose_info()

            # 2. 路径规划
            pre_shipToNextWPDistance = 0
            
            nav_context = self._update_navigation_target(pose_info)
            shipToNextWPDistance =  nav_context['shipToNextWPDistance']
            heading = nav_context['degreeAship']
            nextPointIndex = self.routePlaneService.curNextIndex
            prevPointIndex = self.routePlaneService.curPrevIndex
            if shipToNextWPDistance == pre_shipToNextWPDistance:
                self.isbug +=1
            pre_shipToNextWPDistance = shipToNextWPDistance
            
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
            

            if self.destPointIndex == -1:
                """无导航目标点"""
                return


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
            action= self.ppo_agent.run(state,self.reward,self.done or self.arrive,global_step,self.episode_reward_sum,e)
            #=============================在SAC里做正则，因为这里的state的后继维度还有用
            print("===========time",time)
            
            adviseSpeed = action[0,0]*100
            adviseRotate = action[0,1]*100
            self.next_state, self.reward, advisedHeading, =\
                self.step(state.tolist(),action,laser_scan,heading, shipToNextWPDistance)
            self.episode_reward_sum += self.reward
            
            self.ppo_agent.anneal_lr(self.episode_reward_sum)         #依据奖励调整学习率
            
            """算法输出结果保存到global_data"""
            self.global_data.updateAlgorithmOutput(episode, step, int(self.episode_reward_sum), self.reward, MAX_EPOCH, 2)
            self.global_data.updateThrottleRudderOutput(adviseSpeed, adviseRotate, advisedHeading, nextPointIndex,
                                                        shipToNextWPDistance)
            print(f"Global_step:{global_step},Step: {step}, Action:{action},SPEED: {adviseSpeed},ROTATE:{adviseRotate} Reward: {self.reward}, Done: {self.done}")

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
            obstacle_r = 0
        
        reward = heading_r+speed_r-obstacle_r     #朝向+速度（较小）-障碍距离
        print("=======total_r",reward)
        if self.arrive:
            LogUtil.info("Goal!!")
            reward += 100
        elif self.done:
            LogUtil.info("Collision!!")
            reward += -50
        return reward
    
    
    
    
    def normalize_feature(self,feat: np.ndarray):
        """逐样本 0均值1方差规范化 (Z-Score)"""
        # 计算均值和标准差，利用 keepdims=True 保持维度以便广播相减/除
        mean = np.mean(feat, axis=-1, keepdims=True)
        std = np.std(feat, axis=-1, keepdims=True) + 1e-6  # 防除零
        
        feat_norm = (feat - mean) / std
        return feat_norm
    def binary_cross_entropy_v1(self,y_true, y_pred):
        """计算二分类交叉熵损失的三次方"""
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
    
    
    
    def _check_route_update(self):
        route_update_time = self.global_data.device_data.route_version
        cached_route = self.global_data.route
        cached_route_version = getattr(cached_route, "version", 0.0)

        if route_update_time != cached_route_version:
            LogUtil.info(
                f"[route-check] device_route_version={route_update_time} "
                f"cached_route_version={cached_route_version}, reloading route"
            )
            route = self.ros_ctrl.getRoute()
            self.global_data.updateRouteInfo(routeUpdateTime=route.version, route=route)
            self.__reloadNavigationRoute(route)
            self.route_version_checked = route.version
            self._log_route_snapshot("reloaded", route)
        elif cached_route_version != self.route_version_checked:
            self.route_version_checked = cached_route_version
            self._log_route_snapshot("cached-update", cached_route)
    def _log_route_snapshot(self, tag: str, route):
        points = getattr(route, "points", [])
        route_version = getattr(route, "version", 0.0)
        start_index = getattr(route, "start_index", -1)
        if len(points) == 0:
            LogUtil.info(f"[route:{tag}] version={route_version} start={start_index} points=0")
            return

        first_point = points[0]
        LogUtil.info(
            f"[route:{tag}] version={route_version} start={start_index} points={len(points)} "
            f"first=({first_point.lng:.6f},{first_point.lat:.6f})"
        )