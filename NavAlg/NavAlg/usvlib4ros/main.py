import time
import sys
import os
import json
import argparse

# 将父目录添加到 sys.path，确保能正确导入 usvlib4ros 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from usvlib4ros.navigation.usv_ros2_controller import Ros2Controller
from usvlib4ros.navigation.usv_ros2_controller import Ros2Controller
from usvlib4ros import GlobalData
from usvlib4ros import USVAutoNavigationService
from usvlib4ros.user.SAC_NAV import SAC_NAV


def load_config():
    """加载配置文件，优先从同级目录 config.json，其次从项目根目录"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_paths = [
        os.path.join(script_dir, 'config.json'),                      # usvlib4ros/config.json
        os.path.join(os.path.dirname(script_dir), 'config.json'),     # NavAlg/config.json
    ]
    for config_path in config_paths:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    raise FileNotFoundError(f"配置文件未找到，已尝试路径: {config_paths}")


def parse_args():
    parser = argparse.ArgumentParser(description="USV navigation runner")
    parser.add_argument("--host", help="ROS bridge host")
    parser.add_argument("--port", type=int, help="ROS bridge port")
    parser.add_argument("--device-id", dest="device_id", help="USV deviceId")
    return parser.parse_args()


class USVNavMain:

    @classmethod
    def start(cls,host,port,deviceId):
        globalData = GlobalData().getInstance()
        rosCtrl = Ros2Controller(host=host, port=port, deviceId=deviceId, globalData=globalData)
        # navigationService = USVAutoNavigationService(rosCtrl=rosCtrl, globalData=globalData)
        # navigationService.startService()
        nav = SAC_NAV(ros_ctrl=rosCtrl,global_data=globalData)
        nav.startService()
        pass
    pass

if __name__ == '__main__':
    args = parse_args()
    config = load_config().get("ros2", {})
    host = args.host or config.get("host")
    port = args.port or config.get("port", 9090)
    device_id = args.device_id or config.get("deviceId")

    if not host or not device_id:
        raise ValueError("缺少必要配置: host/deviceId，请检查 config.json 或命令行参数")

    USVNavMain.start(host, port, device_id)

    while True:
        time.sleep(1)
