"""
Usage:
(robodiff)$ python eval_real_robot.py -i <ckpt_path> -o <save_dir> --robot_ip <ip_of_ur5>

================ Human in control ==============
Robot movement:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start evaluation (hand control over to policy).
Press "Q" to exit program.

================ Policy in control ==============
Make sure you can hit the robot hardware emergency-stop button quickly! 

Recording control:
Press "S" to stop evaluation and gain control back.
"""

# %%
import time
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
import torch
import dill
import hydra
import pathlib
import skvideo.io
from omegaconf import OmegaConf
import scipy.spatial.transform as st
from diffusion_policy.real_world.real_env_kuavo import KuavoEnv, FakeRobot
# from diffusion_policy.real_world.spacemouse_shared_memory import Spacemouse
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.real_inference_util import (
    get_real_obs_resolution, 
    get_real_obs_dict)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.cv2_util import get_image_transform

input="/app/data/outputs/2024.09.22/07.35.36_train_diffusion_unet_hybrid_pusht_image/checkpoints/latest.ckpt"
output="/app/data/outputs/2024.09.22/07.35.36_train_diffusion_unet_hybrid_pusht_image/checkpoints/output"
robot_ip="192.168.0.204"
match_dataset="/app/data/pusht_real/real_pusht_20230105"
match_episode=None
vis_camera_idx=0
init_joints=False
steps_per_inference=6
max_duration=60
frequency=10
command_latency=0.01
import threading
fake_robot = FakeRobot()

t = threading.Thread(target=fake_robot.run)
t.start()
# event = threading.Event()
with KuavoEnv(
    output_dir=output,
    frequency=frequency,
    n_obs_steps=2,
    obs_image_resolution=(640, 480),
    max_obs_buffer_size=30,
    robot_publish_rate=125,
    video_capture_fps=30,
    video_capture_resolution=(1280, 720),
    ) as env:
    # fake_obs=env.get_fake_obs()
#     while not env.is_ready:
#         print(env.is_ready)
#     print("Ready")
    # real_obs=env.get_obs()
    while 1:
        print(env.is_ready)