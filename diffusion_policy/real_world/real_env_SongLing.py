import rospy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image, JointState
import numpy as np
import cv2
from collections import deque
from scipy.spatial.transform import Rotation as R
from typing import List, Optional, Union, Dict, Callable
import math
import time
from tqdm import tqdm  
import matplotlib.pyplot as plt
import pathlib
DEFAULT_OBS_KEY_MAP = {
    "img":{
        "img01": "/camera/color/image_raw",
        "img02": "/camera_r/color/image_raw",
    },
    "low_dim":{
        "agent_pos":"/puppet/joint_right",
        # "action":"/master/joint_right",
    }
}

DEFAULT_ACT_KEY_MAP = {
    "action":"/master/joint_right",
}

class ObsBuffer:
    def __init__(self, img_buffer_size: int = 30, robot_state_buffer_size: int = 200, obs_key_map: Optional[Dict[str, Dict[str, str]]] = None) -> None:
        self.img_buffer_size = img_buffer_size
        self.robot_state_buffer_size = robot_state_buffer_size
        self.obs_key_map = obs_key_map if obs_key_map is not None else DEFAULT_OBS_KEY_MAP
        
        self.obs_buffer_data = {key: {"data": deque(maxlen=img_buffer_size),"timestamp": deque(maxlen=img_buffer_size),} \
                                for key in self.obs_key_map["img"]}
        
        self.obs_buffer_data.update({key: {"data": deque(maxlen=robot_state_buffer_size),"timestamp": deque(maxlen=robot_state_buffer_size),} \
                                    for key in self.obs_key_map["low_dim"]})
     
        # Subscribe to the ROS topics
        self.img01_suber = rospy.Subscriber(self.obs_key_map["img"]["img01"],Image,lambda msg: self.image_callback(msg, "img01"),)
        self.img02_suber = rospy.Subscriber(self.obs_key_map["img"]["img02"],Image,lambda msg: self.image_callback(msg, "img02"),)
        
        self.agent_pos_suber = rospy.Subscriber(self.obs_key_map["low_dim"]["agent_pos"],JointState,lambda msg: self.joint_callback(msg, "agent_pos"),)
        # self.action_suber = rospy.Subscriber(self.obs_key_map["low_dim"]["action"],JointState,lambda msg: self.joint_callback(msg, "action"),)

    def image_callback(self, msg: Image, key: str):      
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_img = np_arr.reshape((480, 640, 3))  
        resized_img = cv2.resize(cv_img, (256, 256))
        
        self.obs_buffer_data[key]["data"].append(resized_img)
        self.obs_buffer_data[key]["timestamp"].append(msg.header.stamp.to_sec())


    def joint_callback(self, msg: JointState, key: str):
        joint = (msg.position)[:7]
        self.obs_buffer_data[key]["data"].append(joint)
        self.obs_buffer_data[key]["timestamp"].append(msg.header.stamp.to_sec())

    def obs_buffer_is_ready(self):
        return all([len(self.obs_buffer_data[key]["data"]) == self.img_buffer_size for key in self.obs_key_map["img"]]) and \
               all([len(self.obs_buffer_data[key]["data"]) == self.robot_state_buffer_size for key in self.obs_key_map["low_dim"]])

    def stop_subscribers(self):
        self.img01_suber.unregister()
        self.img02_suber.unregister()
        self.agent_pos_suber.unregister()
        self.action_suber.unregister()

    def get_lastest_k_img(self, k: int) -> Dict[int, Dict[str, np.ndarray]]:
        out = {}
        for i, key in enumerate(self.obs_key_map["img"]):
            out[i] = {
                "color": np.array(list(self.obs_buffer_data[key]["data"])[-k:]),
                "timestamp": np.array(list(self.obs_buffer_data[key]["timestamp"])[-k:]),
            }
        return out

    def get_latest_k_robotstate(self, k: int) -> dict:
        out = {}
        for i, key in enumerate(self.obs_key_map["low_dim"]):
            out[key] = {
                "data": np.array(list(self.obs_buffer_data[key]["data"])[-k:]),
                "robot_receive_timestamp": np.array(list(self.obs_buffer_data[key]["timestamp"])[-k:]),
            }
        return out
    
    def wait_buffer_ready(self):
        progress_bars = {}
        position = 0
        for key in self.obs_key_map["img"]:
            progress_bars[key] = tqdm(total=self.img_buffer_size, desc=f"Filling {key}", position=position, leave=True)
            position += 1

        for key in self.obs_key_map["low_dim"]:
            progress_bars[key] = tqdm(total=self.robot_state_buffer_size, desc=f"Filling {key}", position=position, leave=True)
            position += 1


        while not self.obs_buffer_is_ready():
            for key in self.obs_key_map["img"]:
                current_len = len(self.obs_buffer_data[key]["data"])
                progress_bars[key].n = current_len
                progress_bars[key].refresh()

            for key in self.obs_key_map["low_dim"]:
                current_len = len(self.obs_buffer_data[key]["data"])
                progress_bars[key].n = current_len
                progress_bars[key].refresh()

            time.sleep(0.1)  
            
        # 强制将所有进度条填满
        for key in self.obs_key_map["img"]:
            progress_bars[key].n = self.img_buffer_size
            progress_bars[key].refresh()

        for key in self.obs_key_map["low_dim"]:
            progress_bars[key].n = self.robot_state_buffer_size
            progress_bars[key].refresh()
  
        for bar in progress_bars.values():
            bar.close()
      
        for key in self.obs_key_map["img"]:
            print(f"{key} buffer size = {len(self.obs_buffer_data[key]['data'])}")
        for key in self.obs_key_map["low_dim"]:
            print(f"{key} buffer size = {len(self.obs_buffer_data[key]['data'])}")
            
        print("All buffers are ready!")
        time.sleep(0.5)
        
class SongLingActor:
    def __init__(self, act_key_map: Optional[Dict[str, Dict[str, str]]] = None, ):
        self.act_key_map = act_key_map if act_key_map is not None else DEFAULT_ACT_KEY_MAP
        self.target_pub = rospy.Publisher(
            self.act_key_map["action"], 
            JointState, 
            queue_size=10
        )

    def publish_target_pose(self, pose: np.ndarray):
        msg = JointState()
        msg.position = pose.tolist()  # 假设你想要传递位置
        msg.velocity = [0] * 7
        msg.header.stamp = rospy.Time.now()  # 添加时间戳
        msg.name = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]  # 自定义关节名称
        self.target_pub.publish(msg)
        rospy.loginfo("Publishing target pose: %s", msg.position)    


class SongLingEnv:
    def __init__(self, 
                 frequency:int = 10, 
                 n_obs_steps:int = 2, 
                 video_capture_fps=30,
                 robot_publish_rate=200,
                 img_buffer_size = 30,
                 robot_state_buffer_size = 200,
                 obs_key_map: Optional[Dict[str, Dict[str, str]]] = None,
                 video_capture_resolution=(640, 480), # (W,H)
                 output_dir: str = "output",
                 ) -> None:
        assert frequency <= video_capture_fps
        output_dir = pathlib.Path(output_dir)
        assert output_dir.parent.is_dir()
        
        
        self.frequency = frequency
        self.n_obs_steps = n_obs_steps
        self.video_capture_fps = video_capture_fps
        self.robot_publish_rate = robot_publish_rate
        self.img_buffer_size = img_buffer_size
        self.robot_state_buffer_size = robot_state_buffer_size
        self.video_capture_resolution = video_capture_resolution
        self.obs_key_map = obs_key_map if obs_key_map is not None else DEFAULT_OBS_KEY_MAP
        
        
        self.obs_buffer = ObsBuffer(img_buffer_size=self.img_buffer_size, robot_state_buffer_size=self.robot_state_buffer_size, obs_key_map=DEFAULT_OBS_KEY_MAP)
        self.target_publisher = SongLingActor()
    def reset(self):
        self.obs_buffer = ObsBuffer(img_buffer_size=30, robot_state_buffer_size=200, obs_key_map=DEFAULT_OBS_KEY_MAP)
        
    def step(self):
        pass
    def __enter__(self):
        # 进入上下文管理器时的初始化工作
        # 返回 self，或需要绑定的任何其他对象
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 退出上下文时的清理工作
        # 可选：处理任何异常，返回 True 表示 suppress 异常
        pass
    def exec_actions(
        self,
        actions: np.ndarray,
    ):  
        # actions: (T, D) == (T, 6 + 1)
        # assert self.is_ready
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)

        # convert action to pose
        new_actions = actions
        for i in range(len(new_actions)):
            self.target_publisher.publish_target_pose(new_actions[i])
            time.sleep(0.2)
    
    
    def close(self):
        self.obs_buffer.stop_subscribers()
        self.target_publisher.target_pub.unregister()
    
    def is_ready(self):
        return self.obs_buffer.obs_buffer_is_ready()
    
    def get_obs(self) -> dict:
        "observation dict"
        assert self.is_ready

        # get data
        # 30 Hz, camera_receive_timestamp
        k_image = math.ceil(self.n_obs_steps * (self.video_capture_fps / self.frequency))

        """
        Return order T,H,W,C
        {
            0: {
                'color': (T,H,W,C),
                'timestamp': (T,)
            },
            1: ...
        }
        """
        self.last_realsense_data = self.obs_buffer.get_lastest_k_img(k_image)
        
        
        
        """
        Return order T,D
        {
            0: {
                'data': (T,D),
                'robot_receive_timestamp': (T,)
            },
            1: ...
        }
        """
        k_robot = math.ceil(self.n_obs_steps * (self.robot_publish_rate / self.frequency))
        last_robot_data = self.obs_buffer.get_latest_k_robotstate(k_robot)
        # both have more than n_obs_steps data
        
        # align camera obs timestamps
        dt = 1 / self.frequency
        last_timestamp = np.min(
            [x["timestamp"][-2] for x in self.last_realsense_data.values()]
        )
        obs_align_timestamps = last_timestamp - (np.arange(self.n_obs_steps)[::-1] * dt)

        camera_obs = dict()
        camera_obs_timestamps = dict()
        for camera_idx, value in self.last_realsense_data.items():
            this_timestamps = value["timestamp"]
            this_idxs = list()
            for t in obs_align_timestamps:
                this_idx = np.argmin(np.abs(this_timestamps - t))
                # is_before_idxs = np.nonzero(this_timestamps <= t)[0]
                # this_idx = 0
                # if len(is_before_idxs) > 0:
                #     this_idx = is_before_idxs[-1]
                this_idxs.append(this_idx)
            # remap key
            # camera_obs[f"camera_{camera_idx}"] = value["color"][this_idxs]
            # camera_obs_timestamps[f"camera_{camera_idx}"] = this_timestamps[this_idxs]
            camera_obs[f"img0{camera_idx+1}"] = value["color"][this_idxs]
            camera_obs_timestamps[f"img0{camera_idx+1}"] = this_timestamps[this_idxs]
        # align robot obs timestamps
        robot_obs = dict()
        robot_obs_timestamps = dict()
        for robot_state_name, robot_state_data in last_robot_data.items():
            if robot_state_name in self.obs_key_map["low_dim"]:
                this_timestamps = robot_state_data['robot_receive_timestamp']
                this_idxs = list()
                for t in obs_align_timestamps:
                    this_idx = np.argmin(np.abs(this_timestamps - t))
                    # is_before_idxs = np.nonzero(this_timestamps <= t)[0]
                    # this_idx = 0
                    # if len(is_before_idxs) > 0:
                    #     this_idx = is_before_idxs[-1]
                    this_idxs.append(this_idx)
                robot_obs[f"robot_state_{robot_state_name}"] = robot_state_data['data'][this_idxs]
                robot_obs_timestamps[f"robot_state_{robot_state_name}"] = this_timestamps[this_idxs]
        
        

        # return obs
        obs_data = dict(camera_obs)
    
        robot_final_obs = dict()
        robot_final_obs["agent_pos"] = robot_obs["robot_state_agent_pos"]
        
        obs_data.update(robot_final_obs)
        obs_data["timestamp"] = obs_align_timestamps
        
        return obs_data, camera_obs, camera_obs_timestamps, robot_obs, robot_obs_timestamps

    def check_timestamps_diff(self, check_steps=50):
        all_delta_cam0101_cam0201 = []
        all_delta_cam0102_cam0202 = []
        all_delta_cam0101_cam0102 = []
        all_delta_cam0201_cam0202 = []
        
        all_delta_cam0101_rob0101 = []
        all_delta_cam0102_rob0102 = []
        all_delta_rob0101_rob0102 = []
        
  
        for _ in range(check_steps):
            obs_data, camera_obs, camera_obs_timestamps, robot_obs, robot_obs_timestamps = env.get_obs()
            
            # should dt(1/frequency) diff
            delta_cam0101_cam0102 = abs(camera_obs_timestamps["camera_0"][0] - camera_obs_timestamps["camera_0"][1])
            delta_cam0201_cam0202 = abs(camera_obs_timestamps["camera_1"][0] - camera_obs_timestamps["camera_1"][1])
            delta_rob0101_rob0102 = abs(robot_obs_timestamps["robot_state_agent_pos"][0] - robot_obs_timestamps["robot_state_agent_pos"][1])
            all_delta_cam0101_cam0102.append(delta_cam0101_cam0102)
            all_delta_cam0201_cam0202.append(delta_cam0201_cam0202)
            all_delta_rob0101_rob0102.append(delta_rob0101_rob0102)
            
            # should 0 diff
            delta_cam0101_cam0201 = abs(camera_obs_timestamps["camera_0"][0] - camera_obs_timestamps["camera_1"][0])
            delta_cam0102_cam0202= abs(camera_obs_timestamps["camera_0"][1] - camera_obs_timestamps["camera_1"][1])
            delta_cam0101_rob0101 = abs(camera_obs_timestamps["camera_0"][0] - robot_obs_timestamps["robot_state_agent_pos"][0])
            delta_cam0102_rob0102 = abs(camera_obs_timestamps["camera_0"][1] - robot_obs_timestamps["robot_state_agent_pos"][1])
            all_delta_cam0101_cam0201.append(delta_cam0101_cam0201)
            all_delta_cam0102_cam0202.append(delta_cam0102_cam0202)
            all_delta_cam0101_rob0101.append(delta_cam0101_rob0101)
            all_delta_cam0102_rob0102.append(delta_cam0102_rob0102)
            
            time.sleep(0.1)

            
        # plot the diff between the timestamps
        

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))

        # 在第一个子图上绘制前四个差值
        ax1.plot(all_delta_cam0101_cam0201, label="all_delta_cam0101_cam0201")  # 0
        ax1.plot(all_delta_cam0102_cam0202, label="all_delta_cam0102_cam0202")  # 0
        ax1.plot(all_delta_cam0102_rob0102, label="all_delta_cam0102_rob0102")  # 0
        ax1.plot(all_delta_cam0101_rob0101, label="all_delta_cam0101_rob0101")  # 0
        ax1.set_title("should 0 Differences")  # 设置标题
        ax1.legend()  # 显示图例

        # 在第二个子图上绘制后三个差值
        ax2.plot(all_delta_rob0101_rob0102, label="all_delta_rob0101_rob0102")  # 0.1
        ax2.plot(all_delta_cam0101_cam0102, label="all_delta_cam0101_cam0102")  # 0.1
        ax2.plot(all_delta_cam0201_cam0202, label="all_delta_cam0201_cam0202")  # 0.1
        ax2.set_title("should 0.1 Differences")  # 设置标题
        ax2.legend()  # 显示图例

        # 保存图像
        # fig.savefig("min_agrmin_fu2.png")
        fig.savefig("min_agrmin.png")   # good
        # fig.savefig("max_agrmin.png")
        # fig.savefig("max_before.png")
        
        # # show 
        # plt.show()
        
        print(obs_data.keys())

    def save_img_video(self, check_steps=50):
        img_forder = "imgs"
        import os
        if not os.path.exists(img_forder):
            os.makedirs(img_forder)
        for i in range(check_steps):
            obs_data, camera_obs, camera_obs_timestamps, robot_obs, robot_obs_timestamps = env.get_obs()
            for key, value in camera_obs.items():
                for j in range(value.shape[0]):
                    # the img is RGB, should record all check_steps in forder
                    img = value[j][:, :, ::-1]
                    cv2.imwrite(f"{img_forder}/{key}_{j}_{i}.png", img)
            time.sleep(0.1)
                    
                    
if __name__ == "__main__":
    rospy.init_node("test")
    env = SongLingEnv(img_buffer_size=15, robot_state_buffer_size=100)
    print("waiting for the obs buffer to be ready ......")
    env.obs_buffer.wait_buffer_ready()
    # env.check_timestamps_diff(check_steps=50)
    # env.save_img_video(check_steps=20)
    running = True

    while True:
        # command = input("Enter command (s: start, p: pause, q: exit): ")
        # if command == 's':
        #     running = True
        #     print("Started!")
        # elif command == 'p':
        #     running = False
        #     print("Paused!")
        # elif command == 'q':
        #     print("Exiting...")
        #     break

        if running:
            cur_obs, _, _, _, _ = env.get_obs()
            action = cur_obs["agent_pos"]
            env.exec_actions(actions=action)
    env.close()





