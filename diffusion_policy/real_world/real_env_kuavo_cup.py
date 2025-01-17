from typing import Optional
import pathlib
import numpy as np
import time
import shutil
import math
from sensor_msgs.msg import JointState

# from diffusion_policy.real_world.multi_camera_visualizer import MultiCameraVisualizer
# from diffusion_policy.common.replay_buffer import ReplayBuffer
# from diffusion_policy.common.cv2_util import get_image_transform, optimal_row_cols

# =============================================================================
import rospy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32MultiArray

import cv2
from collections import deque
from scipy.spatial.transform import Rotation as R
from typing import List, Optional, Union, Dict, Callable
from tqdm import tqdm  
import matplotlib.pyplot as plt
import signal
import sys
from PIL import Image
from matplotlib.animation import FuncAnimation

from dynamic_biped.msg import robotArmQVVD, recordArmHandPose, robotHandPosition, robot_hand_eff
from dynamic_biped.srv import controlEndHand, controlEndHandRequest, controlEndHandResponse


# =============================================================================

DEFAULT_OBS_KEY_MAP = {
    "img":{
        "img01": {
            "topic":"/cam_1/color/image_raw/compressed",
            "msg_type":CompressedImage,
            },
        "img02": {
            "topic":"/cam_2/color/image_raw/compressed",
            "msg_type":CompressedImage,
            },
        "img03": {
            "topic":"/cam_3/color/compressed",
            "msg_type":CompressedImage,
            }
    },
    "low_dim":{
        # "cmd_eef": {
        #     "topic":"/drake_ik/cmd_arm_hand_pose", 
        #     "msg_type":recordArmHandPose,
        #     },
        # "state_eef": {
        #     "topic":"/drake_ik/real_arm_hand_pose",
        #     "msg_type":recordArmHandPose,
        #     },
        
        # "cmd_joint": {
        #     "topic":"/kuavo_arm_traj",
        #     "msg_type":JointState 
        #     },
        "state_joint": {
            "topic":"/robot_arm_q_v_tau",
            "msg_type":robotArmQVVD,
            },
        
        # "cmd_gripper": {
        #     "topic":"/robot_hand_eff",
        #     "msg_type":robot_hand_eff,
        #     },
        "state_gripper": {
            "topic":"/robot_hand_position",
            "msg_type":robotHandPosition,
            },
    },
}

DEFAULT_ACT_KEY_MAP = {
    # "target_left_eef_pose": "/drake_ik/target_LHandEef",
    "taget_gripper": "/control_end_hand",
    "target_joint": "/kuavo_arm_traj",
}

# GRIPPER_OPEN_STATE = "[0, 30, 0, 0, 0, 0, 0, 30, 0, 0, 0, 0]"
# HAND_OPEN_STATE = "[0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]"
HAND_OPEN_STATE = "[0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]"
# GRIPPER_CLOSE_STATE = "[30, 30, 90, 90, 90, 90, 30, 30, 90, 90, 90, 90]"
# HAND_CLOSE_STATE = "[59, 99, 32, 44, 51, 50, 0, 0, 0, 0, 0, 0]"
HAND_CLOSE_STATE = "[0, 99, 45, 50, 53, 40, 0, 0, 0, 0, 0, 0]"
    
class ObsBuffer:
    def __init__(self, img_buffer_size: int = 30, robot_state_buffer_size: int = 100, obs_key_map: Optional[Dict[str, Dict[str, str]]] = None) -> None:
        self.img_buffer_size = img_buffer_size
        self.robot_state_buffer_size = robot_state_buffer_size
        self.obs_key_map = obs_key_map if obs_key_map is not None else DEFAULT_OBS_KEY_MAP
        
        
        self.obs_buffer_data = {key: {"data": deque(maxlen=img_buffer_size),"timestamp": deque(maxlen=img_buffer_size),} \
                                for key in self.obs_key_map["img"]}
        
        self.obs_buffer_data.update({key: {"data": deque(maxlen=robot_state_buffer_size),"timestamp": deque(maxlen=robot_state_buffer_size),} \
                                    for key in self.obs_key_map["low_dim"]})
       
        self.callback_key_map = {
            CompressedImage: self.compressedImage_callback,
            recordArmHandPose: self.recordArmHandPose_callback,
            JointState: self.joint_callback,
            robotArmQVVD: self.robotArmQVVD_callback,
            robot_hand_eff: self.robot_hand_eff_callback,
            robotHandPosition: self.robotHandPosition_callback
        }

        self.suber_dict = {}

        self.setup_subscribers()
    
    
    # Subscribe to the ROS topics
    def compressedImage_callback(self, msg: CompressedImage, key: str):
        np_arr = np.frombuffer(msg.data, np.uint8)

        cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(cv_img, (384, 384))
        self.obs_buffer_data[key]["data"].append(resized_img)
        self.obs_buffer_data[key]["timestamp"].append(msg.header.stamp.to_sec())
    # def left_control_hand(self, msg: robotHandPosition, key: str):
    #     left_hand_pose = msg.left_hand_position
    #     if left_hand_pose[-1] == 0:
    #         grip = 0
    #     elif left_hand_pose[-1] == 90:
    #         grip = 1
    #     else:
    #         print("hand pose error")
    #     self.obs_buffer_data[key]["data"].append(grip)
    #     self.obs_buffer_data[key]["timestamp"].append(msg.header.stamp.to_sec())
    def recordArmHandPose_callback(self, msg: recordArmHandPose, key: str):
        # Float64Array ()
        xyz = np.array(msg.left_pose.pos_xyz)
        xyzw = np.array(msg.left_pose.quat_xyzw)
        rotation = R.from_quat(xyzw)
        euler_angles = rotation.as_euler("xyz")
        xyzrpy = np.concatenate((xyz, euler_angles))
        self.obs_buffer_data[key]["data"].append(xyzrpy)
        self.obs_buffer_data[key]["timestamp"].append(msg.header.stamp.to_sec())
        
    def joint_callback(self, msg: JointState, key: str):
        # Float64Array ()
        joint = (msg.position)
        if key == "cmd_joint":
            # convert from degree to rad
            joint = [i * math.pi / 180 for i in joint]
        self.obs_buffer_data[key]["data"].append(joint)
        self.obs_buffer_data[key]["timestamp"].append(msg.header.stamp.to_sec())
    
    def robotArmQVVD_callback(self, msg: robotArmQVVD, key: str):
        # Float64Array ()
        joint = msg.q
        self.obs_buffer_data[key]["data"].append(joint)
        self.obs_buffer_data[key]["timestamp"].append(msg.header.stamp.to_sec())
    
    def robot_hand_eff_callback(self, msg: robot_hand_eff, key: str):
        # Float32Array (12)
        joint = msg.data
        self.obs_buffer_data[key]["data"].append(joint)
        self.obs_buffer_data[key]["timestamp"].append(msg.header.stamp.to_sec())
    
    def robotHandPosition_callback(self, msg: robotHandPosition, key: str):
        # Uint8Array (6) + Uint8Array (6)
        joint = msg.left_hand_position + msg.right_hand_position
        joint= [float(i) for i in joint]
        self.obs_buffer_data[key]["data"].append(joint)
        self.obs_buffer_data[key]["timestamp"].append(msg.header.stamp.to_sec())
            
        
        
        
    def create_callback(self, callback, topic_key):
        return lambda msg: callback(msg, topic_key)
    
    
    def setup_subscribers(self):
        for obs_cls, topics in self.obs_key_map.items():
            for topic_key, topic_info in topics.items():
                topic_name = topic_info["topic"]
                msg_type = topic_info["msg_type"]

                callback = self.callback_key_map.get(msg_type)
                if callback:
                    self.suber_dict[topic_key] = rospy.Subscriber(
                        topic_name, msg_type, self.create_callback(callback, topic_key)
                    )
                    print(f"Subscribed to {topic_name} with callback {callback.__name__}")
                else:
                    print(f"No callback found for message type {msg_type}")

    
    def obs_buffer_is_ready(self, just_img: bool = True):
        if not just_img:
            return all([len(self.obs_buffer_data[key]["data"]) == self.img_buffer_size for key in DEFAULT_OBS_KEY_MAP["img"]]) and \
                all([len(self.obs_buffer_data[key]["data"]) == self.robot_state_buffer_size for key in DEFAULT_OBS_KEY_MAP["low_dim"]])
        else:
            return all([len(self.obs_buffer_data[key]["data"]) == self.img_buffer_size for key in DEFAULT_OBS_KEY_MAP["img"]])

    def stop_subscribers(self):
        for key, suber in self.suber_dict.items():
            suber.unregister()

    def get_lastest_k_img(self, k: int) -> Dict[int, Dict[str, np.ndarray]]:
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
        out = {}
        for i, key in enumerate(self.obs_key_map["img"]):
            out[i] = {
                "color": np.array(list(self.obs_buffer_data[key]["data"])[-k:]),
                "timestamp": np.array(list(self.obs_buffer_data[key]["timestamp"])[-k:]),
            }
        return out

    def get_latest_k_robotstate(self, k: int) -> dict:
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
        out = {}
        for i, key in enumerate(self.obs_key_map["low_dim"]):
            out[key] = {
                "data": np.array(list(self.obs_buffer_data[key]["data"])[-k:]),
                "robot_receive_timestamp": np.array(list(self.obs_buffer_data[key]["timestamp"])[-k:]),
            }
        return out
    
    def wait_buffer_ready(self, just_img: bool = True):
        progress_bars = {}
        position = 0
        for key in self.obs_key_map["img"]:
            progress_bars[key] = tqdm(total=self.img_buffer_size, desc=f"Filling {key}", position=position, leave=True)
            position += 1
        if not just_img:
            for key in self.obs_key_map["low_dim"]:
                progress_bars[key] = tqdm(total=self.robot_state_buffer_size, desc=f"Filling {key}", position=position, leave=True)
                position += 1


        while not self.obs_buffer_is_ready(just_img):
            for key in self.obs_key_map["img"]:
                current_len = len(self.obs_buffer_data[key]["data"])
                progress_bars[key].n = current_len
                progress_bars[key].refresh()
                
            if not just_img:
                for key in self.obs_key_map["low_dim"]:
                    current_len = len(self.obs_buffer_data[key]["data"])
                    progress_bars[key].n = current_len
                    progress_bars[key].refresh()

            time.sleep(0.01)  
            
        # # 强制将所有进度条填满
        # for key in self.obs_key_map["img"]:
        #     progress_bars[key].n = self.img_buffer_size
        #     progress_bars[key].refresh()

        # for key in self.obs_key_map["low_dim"]:
        #     progress_bars[key].n = self.robot_state_buffer_size
        #     progress_bars[key].refresh()
  
        # for bar in progress_bars.values():
        #     bar.close()
      
        # for key in self.obs_key_map["img"]:
        #     print(f"{key} buffer size = {len(self.obs_buffer_data[key]['data'])}")
        # for key in self.obs_key_map["low_dim"]:
        #     print(f"{key} buffer size = {len(self.obs_buffer_data[key]['data'])}")
            
        print("All buffers are ready!")
        time.sleep(0.5)
        
class TargetPublisher:
    def __init__(self):
        # self.target_pub = rospy.Publisher(
        #     DEFAULT_ACT_KEY_MAP["target_left_eef_pose"], 
        #     Float32MultiArray, 
        #     queue_size=10
        # )
        self.joint_pub = rospy.Publisher(
            DEFAULT_ACT_KEY_MAP["target_joint"],
            JointState,
            queue_size=10
        )
                
        self.pub_cnt = 0
        self.last_pose = np.zeros(6)

    def publish_target_pose(self, pose: np.ndarray, cur_state: Optional[np.ndarray] = None):
        msg = Float32MultiArray()
        msg.data = pose.tolist()
        self.joint_pub.publish(msg)
        rospy.loginfo("Publishing target pose: %s", msg.data)
        time.sleep(0.1)
        # with open("target_pose.txt", "a") as f:
        #     f.write(str(msg.data) + "\n")
        # print("state  check: ", msg.data)
        # print("action check: ", cur_state)
        # # check the target pose, if diff is too large, do not publish
        # if self.pub_cnt == 0:
        #     self.last_pose = pose
        #     self.pub_cnt += 1

        # # 假设 pose 和 self.last_pose 都是 [x, y, z, roll, pitch, yaw]
        # translation_current = pose[:3]  # 提取位置部分 (x, y, z)
        # translation_last = self.last_pose[:3]

        # rotation_current = pose[3:]  # 提取姿态部分 (roll, pitch, yaw)
        # rotation_last = self.last_pose[3:]

        # cur_state_translation = cur_state[:3]
        # cur_state_rotation = cur_state[3:]
        # # 计算位置的欧几里得距离
        # translation_distance = np.linalg.norm(translation_current - translation_last)

        # # 计算姿态变化 (简单的欧几里得距离，可能需要进一步处理)
        # rotation_distance = np.linalg.norm(rotation_current - rotation_last)
        
        # translation_distance_cur = np.linalg.norm(translation_current - cur_state_translation)
        # rotation_distance_cur = np.linalg.norm(rotation_current - cur_state_rotation)
        # 设置阈值
        # if translation_distance > 0.1 or rotation_distance > 0.1 or translation_distance_cur > 0.2 or rotation_distance_cur > 0.2:
        #     print("Pose has changed significantly!")
        # else:
        #     print("Pose change is negligible.")
        #     self.target_pub.publish(msg)

    def publish_target_joint(self, joint: np.ndarray, cur_state: Optional[np.ndarray] = None):

        joint = np.rad2deg(joint)
        arm_min = [-180, -20, -135, -100, -135, -45, -45, -180, -180, -180, -180, -180, -45, -45]
        arm_max = [30,   135,  135,  100,  135,  45,  45,  180,  180,  180,  180,  180,  45,  45]
        joint_state = JointState()
        positions  = [0 for _ in range(14)]
        velocities = [0 for _ in range(14)]

        for i in range(len(joint)):
            if joint[i] < arm_min[i]:
                joint[i] = arm_min[i]
            elif joint[i] > arm_max[i]:
                joint[i] = arm_max[i]
        positions[0:7] = joint
        print("send_angle:",[round(x,1) for x in joint])
        velocities[0:7] = [0]*7
        joint_state.position = positions
        joint_state.velocity = velocities
        joint_state.header.stamp = rospy.Time.now()
        # print(joint_state)
        self.joint_pub.publish(joint_state)



    def control_hand(self, left_hand_position: List[float], right_hand_position: List[float]):
        hand_positions = controlEndHandRequest()
        hand_positions.left_hand_position = left_hand_position
        hand_positions.right_hand_position = right_hand_position
        try:
            rospy.wait_for_service('/control_end_hand')
            control_end_hand = rospy.ServiceProxy('/control_end_hand', controlEndHand)
            resp = control_end_hand(hand_positions)
            if resp.result:
                # rospy.loginfo("Gripper control successful")
                pass
            else:
                rospy.logwarn("Gripper control failed")
            return resp.result
        except rospy.ROSException as e:
            rospy.logerr("Service call failed: %s" % e)
            return False
        except KeyboardInterrupt:
            rospy.loginfo("Service call interrupted, shutting down.")
            return False

class KuavoEnv:
    def __init__(self,
                frequency:int = 10, 
                n_obs_steps:int = 2, 
                video_capture_fps=30,
                robot_publish_rate=100,
                img_buffer_size = 30,
                robot_state_buffer_size = 100,
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
        self.hand_close_state, self.hand_open_state = HAND_CLOSE_STATE, HAND_OPEN_STATE

        self.obs_buffer = ObsBuffer(img_buffer_size=self.img_buffer_size, robot_state_buffer_size=self.robot_state_buffer_size, obs_key_map=self.obs_key_map)
        self.target_publisher = TargetPublisher()

    # ======== start-stop API =============
    @property
    def is_ready(self):
        return self.obs_buffer.obs_buffer_is_ready()

    def start(self, wait=True):
        print(self.is_ready)

    def stop(self):
        self.obs_buffer.stop_subscribers()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def get_fake_obs(self):
        '''
        img:(T,H,W,C)
        '''
        import datetime
        first_timestamp = datetime.datetime.now().timestamp()
        second_timestamp = first_timestamp + 0.1
        return {
            "image": np.random.rand(2, 480, 640, 3),
            "agent_pos": np.random.rand(2, 7),
           
            "timestamp": np.array([first_timestamp, second_timestamp])
        }
        
    # ========= async env API ===========
    def get_obs(self, just_img: bool=False) -> dict:
        "observation dict"
        assert self.is_ready

        # get data
        # 30 Hz, camera_receive_timestamp
        k_image = math.ceil(self.n_obs_steps * (self.video_capture_fps / self.frequency))
        self.last_realsense_data = self.obs_buffer.get_lastest_k_img(k_image)
        
        # both have more than n_obs_steps data

        # align camera obs timestamps
        dt = 1 / self.frequency
        last_timestamp = np.max(
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
                # is_before_idxs = np.nonzero(this_timestamps < t)[0]
                # this_idx = 0
                # if len(is_before_idxs) > 0:
                #     this_idx = is_before_idxs[-1]
                this_idxs.append(this_idx)
            # remap key
            camera_obs[f"img0{camera_idx+1}"] = value["color"][this_idxs]
            camera_obs_timestamps[f"img0{camera_idx+1}"] = this_timestamps[this_idxs]

        if not just_img:
            k_robot = math.ceil(self.n_obs_steps * (self.robot_publish_rate / self.frequency))
            last_robot_data = self.obs_buffer.get_latest_k_robotstate(k_robot)
            # align robot obs timestamps
            robot_obs = dict()
            robot_obs_timestamps = dict()
            for robot_state_name, robot_state_data in last_robot_data.items():
                if robot_state_name in self.obs_key_map["low_dim"]:
                    this_timestamps = robot_state_data['robot_receive_timestamp']
                    this_idxs = list()
                    for t in obs_align_timestamps:
                        this_idx = np.argmin(np.abs(this_timestamps - t))
                        # is_before_idxs = np.nonzero(this_timestamps < t)[0]
                        # this_idx = 0
                        # if len(is_before_idxs) > 0:
                        #     this_idx = is_before_idxs[-1]
                        this_idxs.append(this_idx)
                    robot_obs[f"ROBOT_{robot_state_name}"] = robot_state_data['data'][this_idxs]
                    robot_obs_timestamps[f"ROBOT_{robot_state_name}"] = this_timestamps[this_idxs]

    

        # ==========================================
        # process raw data to standard obs
        # ==========================================
        obs_data = dict(camera_obs)
        
        if not just_img:
            # process binary value hand if training with 0/1 for open/close hand state
            robot_obs["ROBOT_state_gripper"] = np.array([[0] if gripper_state[2] == 0 else [1] for gripper_state in robot_obs["ROBOT_state_gripper"]])
            # robot_obs["ROBOT_cmd_gripper"] = np.array([[0] if gripper_cmd[2] == 0 else [1] for gripper_cmd in robot_obs["ROBOT_state_gripper"]])
            
            robot_final_obs = dict()
            robot_final_obs["agent_pos"] = np.concatenate((robot_obs["ROBOT_state_joint"][:,:7], robot_obs["ROBOT_state_gripper"]), axis=-1)
            # robot_final_obs["agent_pos"] = np.concatenate((robot_obs["ROBOT_state_eef"][:,:6], robot_obs["ROBOT_state_gripper"]), axis=-1)
        
            obs_data.update(robot_final_obs)
        else:
            robot_obs = None
            robot_obs_timestamps = None
        obs_data["timestamp"] = obs_align_timestamps
        
        return obs_data, camera_obs, camera_obs_timestamps, robot_obs, robot_obs_timestamps
    
    def exec_fake_actions(self, actions: np.ndarray):
        # just print the new actions
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)
        print(f"executing actions: {actions.shape}")
        return
    
    def exec_actions(
        self,
        actions: np.ndarray,
        latency: float,
        start_point: time,
        cur_state: Optional[np.ndarray] = None,
    ):  
        # actions: (T, D) == (T, 6 + 1)
        # assert self.is_ready
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)

        # convert action to pose
        new_actions = actions
        # with open("cur_state.txt", "a") as f:
        #     f.write(str(cur_state) + "\n")
        for i in range(len(new_actions)):
            # self.target_publisher.publish_target_pose(new_actions[i, :-1])
            self.target_publisher.publish_target_joint(new_actions[i, :-1])
            print('==========================================',time.time()-start_point)
            if new_actions[i, -1] > 0.5:
                self.target_publisher.control_hand(left_hand_position=list(map(int, self.hand_close_state[1:-1].split(", ")))[:6], right_hand_position=[0, 0, 0, 0, 0, 0])
            else:
                self.target_publisher.control_hand(left_hand_position=list(map(int, self.hand_open_state[1:-1].split(", ")))[:6], right_hand_position=[0, 0, 0, 0, 0, 0])
            time.sleep(0.1)
        # # record actions
        # if self.action_accumulator is not None:
        #     self.action_accumulator.put(new_actions, new_timestamps)
        # if self.stage_accumulator is not None:
        #     self.stage_accumulator.put(new_stages, new_timestamps)
    
    def check_timestamps_diff(self, check_steps=50):
        all_delta_cam0101_cam0201 = []
        all_delta_cam0102_cam0202 = []
        all_delta_cam0101_cam0102 = []
        all_delta_cam0201_cam0202 = []
        
        all_delta_cam0101_rob0101 = []
        all_delta_cam0102_rob0102 = []
        all_delta_rob0101_rob0102 = []
        
        img_topic = "img01"
        robot_topic = "ROBOT_state_eef"
        for _ in range(check_steps):
            obs_data, camera_obs, camera_obs_timestamps, robot_obs, robot_obs_timestamps = env.get_obs()
            
            # should dt(1/frequency) diff
            delta_cam0101_cam0102 = abs(camera_obs_timestamps[img_topic][0] - camera_obs_timestamps[img_topic][1])
            # delta_cam0201_cam0202 = abs(camera_obs_timestamps["img02"][0] - camera_obs_timestamps["img02"][1])
            delta_rob0101_rob0102 = abs(robot_obs_timestamps[robot_topic][0] - robot_obs_timestamps[robot_topic][1])
            all_delta_cam0101_cam0102.append(delta_cam0101_cam0102)
            # all_delta_cam0201_cam0202.append(delta_cam0201_cam0202)
            all_delta_rob0101_rob0102.append(delta_rob0101_rob0102)
            
            # should 0 diff
            # delta_cam0101_cam0201 = abs(camera_obs_timestamps["img01"][0] - camera_obs_timestamps["img02"][0])
            # delta_cam0102_cam0202= abs(camera_obs_timestamps["img01"][1] - camera_obs_timestamps["img02"][1])
            delta_cam0101_rob0101 = abs(camera_obs_timestamps[img_topic][0] - robot_obs_timestamps[robot_topic][0])
            delta_cam0102_rob0102 = abs(camera_obs_timestamps[img_topic][1] - robot_obs_timestamps[robot_topic][1])
            # all_delta_cam0101_cam0201.append(delta_cam0101_cam0201)
            # all_delta_cam0102_cam0202.append(delta_cam0102_cam0202)
            all_delta_cam0101_rob0101.append(delta_cam0101_rob0101)
            all_delta_cam0102_rob0102.append(delta_cam0102_rob0102)
            
            time.sleep(0.1)

            
        # plot the diff between the timestamps
        
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))

        # 在第一个子图上绘制前四个差值
        # ax1.plot(all_delta_cam0101_cam0201, label="all_delta_cam0101_cam0201")  # 0
        # ax1.plot(all_delta_cam0102_cam0202, label="all_delta_cam0102_cam0202")  # 0
        ax1.plot(all_delta_cam0102_rob0102, label="all_delta_cam0102_rob0102")  # 0
        ax1.plot(all_delta_cam0101_rob0101, label="all_delta_cam0101_rob0101")  # 0
        ax1.set_title("should 0 Differences")  # 设置标题
        ax1.legend()  # 显示图例

        # 在第二个子图上绘制后三个差值
        ax2.plot(all_delta_rob0101_rob0102, label="all_delta_rob0101_rob0102")  # 0.1
        ax2.plot(all_delta_cam0101_cam0102, label="all_delta_cam0101_cam0102")  # 0.1
        # ax2.plot(all_delta_cam0201_cam0202, label="all_delta_cam0201_cam0202")  # 0.1
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
    
    def check_data_accuracy(self, check_steps=50):
        robot_cmd = []
        for _ in range(check_steps):
            obs_data, camera_obs, camera_obs_timestamps, robot_obs, robot_obs_timestamps = env.get_obs()
            time.sleep(0.1)
            robot_cmd.append(obs_data["state"][0])
        fig, ax = plt.subplots(1, len(robot_cmd[0]), figsize=(24, 12))
        robot_cmd = np.array(robot_cmd)
        for i in range(len(robot_cmd[0])):
            ax[i].plot(robot_cmd[:, i])
            ax[i].set_title(f"cmd_{i}")
        plt.show()
        plt.savefig("cmd.png")
    
    def record_video(self, output_video_path, width=640, height=480, fps=30, hstack=True, stop_event=None,title="IL"):
        obs, _, _, _, _ = self.get_obs(just_img=True)
        cam_num = 0
        for k in obs.keys():
            if "img" in k:
                cam_num += 1
        fourcc = cv2.VideoWriter_fourcc(*'H264')  # 使用 H264 编码器
        if hstack:
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (cam_num * width, height))
        else:
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height * cam_num))
            
        while not stop_event.is_set():
            # 获取环境的观察数据
            obs, _, _, _, _ = self.get_obs(just_img=True)
            imgs = []
            for k, v in obs.items():
                if "img" in k:
                    img = obs[k][-1]
                    img = cv2.resize(img, (width, height))
                    img01_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    imgs.append(img01_bgr)
                    
            concatenated_img = np.hstack(imgs)  # 横向拼接图像
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(concatenated_img, title, (50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            # 显示拼接后的图像
            # cv2.imshow('speedx1 head and third perspective cam', concatenated_img)
            # 确保图像窗口能被更新
            # key = cv2.waitKey(1)  # 等待1毫秒，刷新窗口
            
            # 按 'q' 键退出循环
            # if key == ord('q'):
            #     break
            
            # 写入视频文件
            out.write(concatenated_img)
            time.sleep(1/fps)
        out.release()
                    
if __name__ == "__main__":
    try:

        rospy.init_node("test")
        # def handle_control_end_hand(req):
        #     recv_hand_pose = req.left_hand_position + req.right_hand_position
        #     recv_hand_pose = [float(i) for i in recv_hand_pose]
        #     # rospy.loginfo("Received hand_position: %s", recv_hand_pose)   
        #     success = True
        #     return controlEndHandResponse(result=success)
        # rospy.Service('/control_end_hand', controlEndHand, handle_control_end_hand)
        
        
        env = KuavoEnv(img_buffer_size=30, robot_state_buffer_size=100)
        print("waiting for the obs buffer to be ready ......")
        env.obs_buffer.wait_buffer_ready()
        # env.check_timestamps_diff(check_steps=100)
        # env.check_data_accuracy(check_steps=50)
        # env.save_img_video(check_steps=20)
        # 用于确保 MP4 文件正常保存的 flag
        # 初始化视频写入器（设置为适当的分辨率）
        frame_width = 1280  # 合并后的图像宽度（假设每张图像的宽度为640）
        frame_height = 480  # 图像的高度
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 你可以选择其他格式，比如'XVID'
        out = cv2.VideoWriter('record.mp4', fourcc, 30 , (frame_width, frame_height))
        if not out.isOpened():
            print("Error: Video file could not be opened.")
            sys.exit(1)

        # 捕捉退出信号，确保视频保存
        def handle_exit_signal(signum, frame):
            print("Signal received, saving video and cleaning up...")
            if out is not None:
                out.release()  # 释放视频写入器
            sys.exit(0)  # 正常退出

        # 注册信号处理器
        signal.signal(signal.SIGINT, handle_exit_signal)  # Ctrl+C
        signal.signal(signal.SIGQUIT, handle_exit_signal)  # Ctrl+\
 
        while 1:
            cur_obs, _, _, _, _ = env.get_obs()  
            
            for sensor, v in cur_obs.items():
                print(f'{sensor=}', f'{v.shape=}')
            
            img01 = np.array(cur_obs['img01'][-1])  
            img02 = np.array(cur_obs['img02'][-1])  
            img01 = cv2.resize(img01, (640,480))
            img02 = cv2.resize(img02, (640,480))
            
            if img01.shape[0] != img02.shape[0]:
                img02 = cv2.resize(img02, (img02.shape[1], img01.shape[0]))  
            img01_bgr = cv2.cvtColor(img01, cv2.COLOR_RGB2BGR)
            img02_bgr = cv2.cvtColor(img02, cv2.COLOR_RGB2BGR)
            # 水平合并图像
            # 将这两个图像并排显示
            combined_img = np.hstack((img01_bgr, img02_bgr))

            # 显示图像
            cv2.imshow('Image Stream', combined_img)

            # 写入视频流
            out.write(combined_img)

            # 处理按键事件，按'q'退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
       
        out.release()
        cv2.destroyAllWindows()

    except KeyboardInterrupt:
        rospy.loginfo("Shutting down node...")
        rospy.signal_shutdown("Manual shutdown")
        
