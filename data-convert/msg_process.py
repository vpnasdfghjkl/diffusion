import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import math


class ProcessUtil:
    @staticmethod
    def process_compressed_image(msg, resize=None):
        """
        Args:
            msg: CompressedImage
                The compressed image message.
            resize: tuple (width, height), optional
                Resize the image to this size. If None, the original size is used.
        
        Returns:
            Dict:
                - "data" (np.array): Image data with shape (height, width, 3).
                - "timestamp" (float): The timestamp of the image in seconds.
        """
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(cv_img, (resize[0], resize[1])) if resize else cv_img
        return {"data": resized_img, "timestamp": msg.header.stamp.to_sec()}

    @staticmethod
    def process_record_arm_hand_pose(msg):
        """
        args:
            msg: recordArmHandPose
        return:
            left_right_eef: np.array (12,)
        """
        left_right_eef = np.concatenate([
            np.concatenate((
                np.array(pose.pos_xyz), 
                R.from_quat(pose.quat_xyzw).as_euler('xyz')
            ))
            for pose in [msg.left_pose, msg.right_pose]
        ])
        return {"data": left_right_eef, "timestamp": msg.header.stamp.to_sec()}

    @staticmethod
    def process_joint(msg, is_cmd: bool = True):
        """Process joint data to extract joint angles"""
        if is_cmd:
            joint = [i * math.pi / 180 for i in list(msg.position)]
        else:
            joint = msg.q
        return {"data": joint, "timestamp": msg.header.stamp.to_sec()}


    @staticmethod
    def process_hand_data(msg, is_cmd: bool = True, is_binary: bool = False, HAND_CLOSE_STATE:str= "[59, 99, 32, 44, 51, 50, 0, 0, 0, 0, 0, 0]"):
        """Process Dexterous hand data to extract hand state"""
        if is_cmd:
            dex_hand = msg.data
        else:
            dex_hand = [float(i) for i in msg.left_hand_position + msg.right_hand_position]
        if is_binary:
            dex_hand = [
                1 if dex_hand[0] == list(map(int, HAND_CLOSE_STATE[1:-1].split(", ")))[0] else 0,
                1 if dex_hand[0] == list(map(int, HAND_CLOSE_STATE[1:-1].split(", ")))[0] else 0
            ]
        return {"data": dex_hand, "timestamp": msg.header.stamp.to_sec()}
    
   