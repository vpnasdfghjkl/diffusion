from typing import Optional
import pathlib
import numpy as np
import time
import shutil
import math
from sensor_msgs.msg import Image, JointState
from multiprocessing.managers import SharedMemoryManager
# from diffusion_policy.real_world.rtde_interpolation_controller import (
#     RTDEInterpolationController,
# )
# from diffusion_policy.real_world.multi_realsense import MultiRealsense, SingleRealsense
from diffusion_policy.real_world.video_recorder import VideoRecorder
from diffusion_policy.common.timestamp_accumulator import (
    TimestampObsAccumulator,
    TimestampActionAccumulator,
    align_timestamps,
)
# from diffusion_policy.real_world.multi_camera_visualizer import MultiCameraVisualizer
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.cv2_util import get_image_transform, optimal_row_cols


# =============================================================================
import rospy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32MultiArray

import cv2
from collections import deque
from scipy.spatial.transform import Rotation as R
from typing import List, Optional, Union, Dict, Callable


from dynamic_biped.msg import robotArmInfo, recordArmHandPose, robotHandPosition
from dynamic_biped.srv import controlEndHand, controlEndHandRequest


# =============================================================================

DEFAULT_OBS_KEY_MAP = {
    "img":{
        "img01": "/head_camera/color/image_raw/compressed",
    },
    "low_dim":{
        "obs_state_eef_pose": "/drake_ik/real_arm_hand_pose",
        "obs_state_hand": "/robot_hand_position",
        "obs_cmd_eef_pose": "/drake_ik/cmd_arm_hand_pose",
    },
}

DEFAULT_ACT_KEY_MAP = {
    "target_left_eef_pose": "/drake_ik/target_LHandEef",
    "taget_gripper": "/control_end_hand",
}

# GRIPPER_OPEN_STATE = "[0, 30, 0, 0, 0, 0, 0, 30, 0, 0, 0, 0]"
HAND_OPEN_STATE = "[0, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]"
# GRIPPER_CLOSE_STATE = "[30, 30, 90, 90, 90, 90, 30, 30, 90, 90, 90, 90]"
HAND_CLOSE_STATE = "[30, 30, 90, 90, 90, 90, 0, 0, 0, 0, 0, 0]"

    
class ObsBuffer:
    def __init__(self, img_buffer_size: int = 30, robot_state_buffer_size: int = 120):
        self.img_buffer_size = img_buffer_size
        self.robot_state_buffer_size = robot_state_buffer_size
        self.obs_buffer_data = {key: {"data": deque(maxlen=img_buffer_size),"timestamp": deque(maxlen=img_buffer_size),} \
                                for key in self.obs_key_map["img"]}
        
        self.obs_buffer_data.update({key: {"data": deque(maxlen=robot_state_buffer_size),"timestamp": deque(maxlen=robot_state_buffer_size),} \
                                    for key in self.obs_key_map["low_dim"]})
     
        # Subscribe to the ROS topics
        self.img01_suber = rospy.Subscriber(self.obs_key_map["img"]["img01"],Image,lambda msg: self.image_callback(msg, "img01"),)
        
        self.obs_state_hand_suber = rospy.Subscriber(DEFAULT_OBS_KEY_MAP["low_dim"]["obs_state_hand"],robotHandPosition,lambda msg: self.left_control_hand(msg, "obs_state_hand"),)

        self.obs_state_eef_pose_suber = rospy.Subscriber(DEFAULT_OBS_KEY_MAP["low_dim"]["obs_state_eef_pose"],recordArmHandPose,lambda msg: self.eef_pose_callback(msg, "obs_state_eef_pose"),)
        self.obs_cmd_eef_pose_suber = rospy.Subscriber(DEFAULT_OBS_KEY_MAP["low_dim"]["obs_cmd_eef_pose"],recordArmHandPose,lambda msg: self.eef_pose_callback(msg, "obs_cmd_eef_pose"),)

    def image_callback(self, msg: CompressedImage, key: str):
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        resized_img = cv2.resize(cv_img, (256, 256))

        self.obs_buffer_data[key]["data"].append(resized_img)
        self.obs_buffer_data[key]["timestamp"].append(msg.header.stamp.to_sec())

    def left_control_hand(self, msg: robotHandPosition, key: str):
        left_hand_pose = msg.left_hand_position
        if left_hand_pose[-1] == 0:
            grip = 0
        elif left_hand_pose[-1] == 90:
            grip = 1
        else:
            print("hand pose error")
        self.obs_buffer_data[key]["data"].append(grip)
        self.obs_buffer_data[key]["timestamp"].append(msg.header.stamp.to_sec())

    def eef_pose_callback(self, msg: recordArmHandPose, key: str):
        xyz = np.array(msg.left_pose.pos_xyz)
        xyzw = np.array(msg.left_pose.quat_xyzw)
        rotation = R.from_quat(xyzw)
        euler_angles = rotation.as_euler("xyz")
        xyzrpy = np.concatenate((xyz, euler_angles))

        self.obs_buffer_data[key]["data"].append(xyzrpy)
        self.obs_buffer_data[key]["timestamp"].append(msg.header.stamp.to_sec())

    def obs_buffer_is_ready(self):
        print(len(self.obs_buffer_data["obs_img01"]["data"]))
        print(len(self.obs_buffer_data["obs_state_hand"]["data"]))
        print(len(self.obs_buffer_data["obs_state_eef_pose"]["data"]))
        print(len(self.obs_buffer_data["obs_cmd_eef_pose"]["timestamp"]))
        return all([len(self.obs_buffer_data[key]["data"]) == self.img_buffer_size for key in DEFAULT_OBS_KEY_MAP if "obs_img" in key]) and \
               all([len(self.obs_buffer_data[key]["data"]) == self.robot_state_buffer_size for key in DEFAULT_OBS_KEY_MAP if "obs_img" not in key])

    def stop_subscribers(self):
        self.obs_img01_suber.unregister()
        self.obs_state_hand_suber.unregister()
        self.obs_state_eef_pose_suber.unregister()
        self.obs_cmd_eef_pose_suber.unregister()

    def get_lastest_k_img(self, k: int) -> Dict[int, Dict[str, np.ndarray]]:
        out = {}
        for i, key in enumerate(DEFAULT_OBS_KEY_MAP):
            if "obs_img" in key:
                out[i] = {
                    "color": np.array(list(self.obs_buffer_data[key]["data"])[-k:]),
                    "timestamp": np.array(list(self.obs_buffer_data[key]["timestamp"])[-k:]),
                }
        return out

    def get_latest_k_robotstate(self, k: int) -> dict:
        out = {}
        for i, key in enumerate(DEFAULT_OBS_KEY_MAP):
            if "obs_img" not in key:
                out[key] = {
                    "data": np.array(list(self.obs_buffer_data[key]["data"])[-k:]),
                    "robot_receive_timestamp": np.array(list(self.obs_buffer_data[key]["timestamp"])[-k:]),
                }
        return out

class TargetPublisher:
    def __init__(self):
        self.target_pub = rospy.Publisher(
            DEFAULT_ACT_KEY_MAP["target_left_eef_pose"], 
            Float32MultiArray, 
            queue_size=10
        )

    def publish_target_pose(self, pose: np.ndarray):
        msg = Float32MultiArray()
        msg.data = pose.tolist()
        self.target_pub.publish(msg)
        rospy.loginfo("Publishing target pose: %s", msg.data)

    def control_hand(self, left_hand_position: List[float], right_hand_position: List[float]):
        hand_positions = controlEndHandRequest()
        hand_positions.left_hand_position = left_hand_position
        hand_positions.right_hand_position = right_hand_position
        try:
            rospy.wait_for_service('/control_end_hand')
            control_end_hand = rospy.ServiceProxy('/control_end_hand', controlEndHand)
            resp = control_end_hand(hand_positions)
            if resp.result:
                rospy.loginfo("Gripper control successful")
            else:
                rospy.logwarn("Gripper control failed")
            return resp.result
        except rospy.ROSException as e:
            rospy.logerr("Service call failed: %s" % e)
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

        self.target_publisher = TargetPublisher()


    # ======== start-stop API =============
    @property
    def is_ready(self):
        return self.obs_buffer.obs_buffer_is_ready()

    def start(self, wait=True):
        print(self.is_ready)
        # self.realsense.start(wait=False)
        # self.robot.start(wait=False)
        # if self.multi_cam_vis is not None:
        #     self.multi_cam_vis.start(wait=False)
        # if wait:
        #     self.start_wait()

    def stop(self):
        self.obs_buffer.stop_subscribers()

    # def start_wait(self):
    #     self.realsense.start_wait()
    #     self.robot.start_wait()
    #     if self.multi_cam_vis is not None:
    #         self.multi_cam_vis.start_wait()

    # def stop_wait(self):
    #     self.robot.stop_wait()
    #     self.realsense.stop_wait()
    #     if self.multi_cam_vis is not None:
    #         self.multi_cam_vis.stop_wait()

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
            # "robot_state_obs_state_eef_pose": np.random.rand(2, 6),
            # "robot_state_obs_cmd_eef_pose": np.random.rand(2, 6),
            "timestamp": np.array([first_timestamp, second_timestamp])
        }
        
    # ========= async env API ===========
    def get_obs(self) -> dict:
        "observation dict"
        assert self.is_ready

        # get data
        # 30 Hz, camera_receive_timestamp
        k = math.ceil(self.n_obs_steps * (self.video_capture_fps / self.frequency))

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
        self.last_realsense_data = self.obs_buffer.get_lastest_k_img(k)
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
        last_robot_data = self.obs_buffer.get_latest_k_robotstate(
            k * (self.robot_publish_rate // self.video_capture_fps)
        )
        # both have more than n_obs_steps data

        # align camera obs timestamps
        dt = 1 / self.frequency
        last_timestamp = np.max(
            [x["timestamp"][-1] for x in self.last_realsense_data.values()]
        )
        obs_align_timestamps = last_timestamp - (np.arange(self.n_obs_steps)[::-1] * dt)

        camera_obs = dict()
        for camera_idx, value in self.last_realsense_data.items():
            this_timestamps = value["timestamp"]
            this_idxs = list()
            for t in obs_align_timestamps:
                is_before_idxs = np.nonzero(this_timestamps < t)[0]
                this_idx = 0
                if len(is_before_idxs) > 0:
                    this_idx = is_before_idxs[-1]
                this_idxs.append(this_idx)
            # remap key
            camera_obs[f"camera_{camera_idx}"] = value["color"][this_idxs]

        # align robot obs timestamps
        robot_obs = dict()
        robot_obs_timestamps = dict()
        for robot_state_name, robot_state_data in last_robot_data.items():
            if robot_state_name in self.obs_topic_key_map:
                this_timestamps = robot_state_data['robot_receive_timestamp']
                this_idxs = list()
                for t in obs_align_timestamps:
                    is_before_idxs = np.nonzero(this_timestamps < t)[0]
                    this_idx = 0
                    if len(is_before_idxs) > 0:
                        this_idx = is_before_idxs[-1]
                    this_idxs.append(this_idx)
                robot_obs[f"robot_state_{robot_state_name}"] = robot_state_data['data'][this_idxs]
                robot_obs_timestamps[f"robot_state_{robot_state_name}"] = this_timestamps[this_idxs]

        # TODO: after 2024/09/23
        # accumulate obs
        if self.obs_accumulator is not None:
            self.obs_accumulator.put(robot_obs, robot_obs_timestamps)

        # return obs
        obs_data = dict(camera_obs)
        
        
        history_hand = robot_obs["robot_state_obs_state_hand"]
        history_eef_pose = robot_obs["robot_state_obs_state_eef_pose"]
        # add the 1dim hand to 6dim eef_pose
        
        robot_final_obs = dict()
        robot_final_obs["state"] = np.concatenate((history_eef_pose, history_hand), axis=1)
        obs_data.update(robot_final_obs)
        obs_data["timestamp"] = obs_align_timestamps
        
        return obs_data
    
    def exec_fake_actions(self, actions: np.ndarray):
        # just print the new actions
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)
        print(f"executing actions: {actions.shape}")
        return
    
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
        self.target_publisher.publish_target_pose(new_actions)
        # schedule waypoints
        for i in range(len(new_actions)):
            self.target_publisher.publish_target_pose(new_actions[i, :6])
            if new_actions[i, -1] > 0.5:
                self.target_publisher.control_hand(left_hand_position=list(map(int, self.hand_close_state[1:-1].split(", ")))[:6], right_hand_position=[0, 0, 0, 0, 0, 0])
            else:
                self.target_publisher.control_hand(left_hand_position=list(map(int, self.hand_open_state[1:-1].split(", ")))[:6], right_hand_position=[0, 0, 0, 0, 0, 0])
        
        # # record actions
        # if self.action_accumulator is not None:
        #     self.action_accumulator.put(new_actions, new_timestamps)
        # if self.stage_accumulator is not None:
        #     self.stage_accumulator.put(new_stages, new_timestamps)

