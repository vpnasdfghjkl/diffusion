from typing import Optional
import pathlib
import numpy as np
import time
import shutil
import math
from multiprocessing.managers import SharedMemoryManager
from diffusion_policy.real_world.rtde_interpolation_controller import RTDEInterpolationController
from diffusion_policy.real_world.multi_realsense import MultiRealsense, SingleRealsense
from diffusion_policy.real_world.video_recorder import VideoRecorder
from diffusion_policy.common.timestamp_accumulator import (
    TimestampObsAccumulator, 
    TimestampActionAccumulator,
    align_timestamps
)
from diffusion_policy.real_world.multi_camera_visualizer import MultiCameraVisualizer
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.cv2_util import (
    get_image_transform, optimal_row_cols)


# =============================================================================
import rospy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32MultiArray

import cv2
from collections import deque
from scipy.spatial.transform import Rotation as R
from typing import List, Optional, Union, Dict, Callable


from dynamic_biped.msg import robotArmInfo,recordArmHandPose,robotHandPosition
# =============================================================================
# DEFAULT_OBS_KEY_MAP = {
#     # robot
#     'ActualTCPPose': 'robot_eef_pose',
#     'ActualTCPSpeed': 'robot_eef_pose_vel',
#     'ActualQ': 'robot_joint',
#     'ActualQd': 'robot_joint_vel',
#     # timestamps
#     'step_idx': 'step_idx',
#     'timestamp': 'timestamp'
# }

DEFAULT_OBS_KEY_MAP = {
    'obs_img01': '/head_camera/color/image_raw/compressed',
    'obs_state_hand': '/robot_hand_position',
    'obs_cmd_eef_pose': '/drake_ik/cmd_arm_hand_pose',
    'obs_state_eef_pose': '/drake_ik/real_arm_hand_pose',
}
DEFAULT_ACT_KEY_MAP = {
    'target_eef_pose': '/drake_ik/target_LHandEef',
    'taget_gripper': '/control_end_hand',
}

class ObsBuffer:
    def __init__(self, buffer_size: int = 30):
        self.buffer_size = buffer_size
        self.data = {key: {'data': deque(maxlen=buffer_size), 'timestamp': deque(maxlen=buffer_size)} for key in DEFAULT_OBS_KEY_MAP}
        
        # Subscribe to the ROS topics
        rospy.Subscriber(DEFAULT_OBS_KEY_MAP['obs_img01'], CompressedImage, lambda msg: self.image_callback(msg, 'obs_img01'))
        rospy.Subscriber(DEFAULT_OBS_KEY_MAP['obs_state_hand'], robotHandPosition, lambda msg: self.hand_callback(msg, 'obs_state_hand'))
        
        rospy.Subscriber(DEFAULT_OBS_KEY_MAP['obs_state_eef_pose'], recordArmHandPose, lambda msg: self.eef_pose_callback(msg, 'obs_state_eef_pose'))
        rospy.Subscriber(DEFAULT_OBS_KEY_MAP['obs_cmd_eef_pose'], recordArmHandPose, lambda msg: self.eef_pose_callback(msg, 'obs_cmd_eef_pose'))

    def image_callback(self, msg: CompressedImage, key: str):
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        resized_img = cv2.resize(cv_img, (256, 256))
        
        self.data[key]['data'].append(resized_img)
        self.data[key]['timestamp'].append(msg.header.stamp.to_sec()) 

    def hand_callback(self, msg: robotHandPosition, key: str):
        left_hand_pose=msg.left_hand_position
        if left_hand_pose[-1]==0:
            grip=0
        elif left_hand_pose[-1]==90:
            grip=1
        else:
            print("hand pose error")
        self.data[key]['data'].append(grip) 
        self.data[key]['timestamp'].append(msg.header.stamp.to_sec())
        
    
    def eef_pose_callback(self, msg: recordArmHandPose, key: str):
        xyz = np.array(msg.left_pose.pos_xyz)
        xyzw = np.array(msg.left_pose.quat_xyzw)
        rotation = R.from_quat(xyzw)
        euler_angles = rotation.as_euler('xyz')
        xyzrpy = np.concatenate((xyz, euler_angles))

        self.data[key]['data'].append(xyzrpy) 
        self.data[key]['timestamp'].append(msg.header.stamp.to_sec())

    def get_lastest_k_img(self, k: int) -> Dict[int, Dict[str, np.ndarray]]:
        out = {}
        for i, key in enumerate(DEFAULT_OBS_KEY_MAP):
            if 'obs_img' in key:
                out[i] = {
                    'color': np.array(self.data[key]['data'][-k:]),  
                    'timestamp': np.array(self.data[key]['timestamp'][-k:])  
                }
        return out

    
    def get_latest_k_robotstate(self, k: int) -> dict:
        out = {}
        for i,key in enumerate(DEFAULT_OBS_KEY_MAP):
            if 'obs_img' not in key:
                out[key] = {
                    'data': np.array(self.data[key]['data'][-k:]),
                    'robot_receive_timestamp': np.array(self.data[key]['timestamp'][-k:])
                }
        return out
    

class KuavoEnv:
    def __init__(self, 
            # required params
            output_dir,
            ROS_MASTER_URI='http://localhost:11311',
            # env params
            frequency=10,
            n_obs_steps=2,
            # obs
            obs_image_resolution=(640,480),
            max_obs_buffer_size=30,
            obs_topic_key_map=DEFAULT_OBS_KEY_MAP,
            act_topic_key_map=DEFAULT_ACT_KEY_MAP,
            
            # camera_serial_numbers=None,
            obs_key_map=DEFAULT_OBS_KEY_MAP,
            obs_float32=False,
            # action
            # max_pos_speed=0.25,
            # max_rot_speed=0.6,
            # robot
            # tcp_offset=0.13,
            # init_joints=False,
            # video capture params
            video_capture_fps=30,
            video_capture_resolution=(1280,720),
            # saving params
            # record_raw_video=True,
            # thread_per_video=2,
            # video_crf=21,
            # vis params
            # enable_multi_cam_vis=True,
            # multi_cam_vis_resolution=(1280,720),
            # shared memory
            # shm_manager=None
            ):
        assert frequency <= video_capture_fps
        output_dir = pathlib.Path(output_dir)
        assert output_dir.parent.is_dir()
        
        video_dir = output_dir.joinpath('videos')
        video_dir.mkdir(parents=True, exist_ok=True)
        
        zarr_path = str(output_dir.joinpath('replay_buffer.zarr').absolute())
        
        replay_buffer = ReplayBuffer.create_from_path(
            zarr_path=zarr_path, mode='a')


        color_tf = get_image_transform(
            input_res=video_capture_resolution,
            output_res=obs_image_resolution, 
            # obs output rgb
            bgr_to_rgb=True)
        color_transform = color_tf
        
        if obs_float32:
            color_transform = lambda x: color_tf(x).astype(np.float32) / 255

        def transform(data):
            data['color'] = color_transform(data['color'])
            return data
    
        recording_transfrom = None
        recording_fps = video_capture_fps
        recording_pix_fmt = 'bgr24'

        self.obs_buffer = ObsBuffer(max_obs_buffer_size)
        

        self.video_capture_fps = video_capture_fps
        self.frequency = frequency
        self.n_obs_steps = n_obs_steps
        self.max_obs_buffer_size = max_obs_buffer_size
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        
        self.obs_topic_key_map = obs_topic_key_map
        self.act_topic_key_map = act_topic_key_map
        
        self.obs_key_map = obs_key_map
        
        # recording
        self.output_dir = output_dir
        self.video_dir = video_dir
        self.replay_buffer = replay_buffer
        # temp memory buffers
        self.last_realsense_data = None
        # recording buffers
        self.obs_accumulator = None
        self.action_accumulator = None
        self.stage_accumulator = None

        self.start_time = None
    
    # ======== start-stop API =============
    @property
    def is_ready(self):
        return self.realsense.is_ready and self.robot.is_ready
    
    def start(self, wait=True):
        self.realsense.start(wait=False)
        self.robot.start(wait=False)
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.start(wait=False)
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        self.end_episode()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.stop(wait=False)
        self.robot.stop(wait=False)
        self.realsense.stop(wait=False)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.realsense.start_wait()
        self.robot.start_wait()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.start_wait()
    
    def stop_wait(self):
        self.robot.stop_wait()
        self.realsense.stop_wait()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.stop_wait()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

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

        last_robot_data = self.obs_buffer.get_latest_k_robotstate(k)
        # both have more than n_obs_steps data

        # align camera obs timestamps
        dt = 1 / self.frequency
        last_timestamp = np.max([x['timestamp'][-1] for x in self.last_realsense_data.values()])
        obs_align_timestamps = last_timestamp - (np.arange(self.n_obs_steps)[::-1] * dt)

        camera_obs = dict()
        for camera_idx, value in self.last_realsense_data.items():
            this_timestamps = value['timestamp']
            this_idxs = list()
            for t in obs_align_timestamps:
                is_before_idxs = np.nonzero(this_timestamps < t)[0]
                this_idx = 0
                if len(is_before_idxs) > 0:
                    this_idx = is_before_idxs[-1]
                this_idxs.append(this_idx)
            # remap key
            camera_obs[f'camera_{camera_idx}'] = value['color'][this_idxs]

        # align robot obs\
        robot_obs = dict()
        for robot_state_name, robot_state_data in last_robot_data.items():
            if robot_state_name in self.obs_topic_key_map:
                robot_state_data = last_robot_data[robot_state_name]['data']
                robot_state_timestamps = last_robot_data[robot_state_name]['robot_receive_timestamp']
                this_timestamps = robot_state_timestamps
                this_idxs = list()
                for t in obs_align_timestamps:
                    is_before_idxs = np.nonzero(this_timestamps < t)[0]
                    this_idx = 0
                    if len(is_before_idxs) > 0:
                        this_idx = is_before_idxs[-1]
                    this_idxs.append(this_idx) 
        robot_obs[robot_state_name] = robot_state_data[this_idxs]
                
        # TODO: after 2024/09/23                      
        # accumulate obs
        if self.obs_accumulator is not None:
            self.obs_accumulator.put(
                robot_obs_raw,
                robot_timestamps
            )

        # return obs
        obs_data = dict(camera_obs)
        obs_data.update(robot_obs)
        obs_data['timestamp'] = obs_align_timestamps
        return obs_data
    
    def exec_actions(self, 
            actions: np.ndarray, 
            timestamps: np.ndarray, 
            stages: Optional[np.ndarray]=None):
        assert self.is_ready
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)
        if not isinstance(timestamps, np.ndarray):
            timestamps = np.array(timestamps)
        if stages is None:
            stages = np.zeros_like(timestamps, dtype=np.int64)
        elif not isinstance(stages, np.ndarray):
            stages = np.array(stages, dtype=np.int64)

        # convert action to pose
        receive_time = time.time()
        is_new = timestamps > receive_time
        new_actions = actions[is_new]
        new_timestamps = timestamps[is_new]
        new_stages = stages[is_new]

        # schedule waypoints
        for i in range(len(new_actions)):
            self.robot.schedule_waypoint(
                pose=new_actions[i],
                target_time=new_timestamps[i]
            )
        
        # record actions
        if self.action_accumulator is not None:
            self.action_accumulator.put(
                new_actions,
                new_timestamps
            )
        if self.stage_accumulator is not None:
            self.stage_accumulator.put(
                new_stages,
                new_timestamps
            )
    
    def get_robot_state(self):
        return self.robot.get_state()

    # recording API
    def start_episode(self, start_time=None):
        "Start recording and return first obs"
        if start_time is None:
            start_time = time.time()
        self.start_time = start_time

        assert self.is_ready

        # prepare recording stuff
        episode_id = self.replay_buffer.n_episodes
        this_video_dir = self.video_dir.joinpath(str(episode_id))
        this_video_dir.mkdir(parents=True, exist_ok=True)
        n_cameras = self.realsense.n_cameras
        video_paths = list()
        for i in range(n_cameras):
            video_paths.append(
                str(this_video_dir.joinpath(f'{i}.mp4').absolute()))
        
        # start recording on realsense
        self.realsense.restart_put(start_time=start_time)
        self.realsense.start_recording(video_path=video_paths, start_time=start_time)

        # create accumulators
        self.obs_accumulator = TimestampObsAccumulator(
            start_time=start_time,
            dt=1/self.frequency
        )
        self.action_accumulator = TimestampActionAccumulator(
            start_time=start_time,
            dt=1/self.frequency
        )
        self.stage_accumulator = TimestampActionAccumulator(
            start_time=start_time,
            dt=1/self.frequency
        )
        print(f'Episode {episode_id} started!')
    
    def end_episode(self):
        "Stop recording"
        assert self.is_ready
        
        # stop video recorder
        self.realsense.stop_recording()

        if self.obs_accumulator is not None:
            # recording
            assert self.action_accumulator is not None
            assert self.stage_accumulator is not None

            # Since the only way to accumulate obs and action is by calling
            # get_obs and exec_actions, which will be in the same thread.
            # We don't need to worry new data come in here.
            obs_data = self.obs_accumulator.data
            obs_timestamps = self.obs_accumulator.timestamps

            actions = self.action_accumulator.actions
            action_timestamps = self.action_accumulator.timestamps
            stages = self.stage_accumulator.actions
            n_steps = min(len(obs_timestamps), len(action_timestamps))
            if n_steps > 0:
                episode = dict()
                episode['timestamp'] = obs_timestamps[:n_steps]
                episode['action'] = actions[:n_steps]
                episode['stage'] = stages[:n_steps]
                for key, value in obs_data.items():
                    episode[key] = value[:n_steps]
                self.replay_buffer.add_episode(episode, compressors='disk')
                episode_id = self.replay_buffer.n_episodes - 1
                print(f'Episode {episode_id} saved!')
            
            self.obs_accumulator = None
            self.action_accumulator = None
            self.stage_accumulator = None

    def drop_episode(self):
        self.end_episode()
        self.replay_buffer.drop_episode()
        episode_id = self.replay_buffer.n_episodes
        this_video_dir = self.video_dir.joinpath(str(episode_id))
        if this_video_dir.exists():
            shutil.rmtree(str(this_video_dir))
        print(f'Episode {episode_id} dropped!')

