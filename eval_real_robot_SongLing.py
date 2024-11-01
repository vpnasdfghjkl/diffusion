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
# from diffusion_policy.real_world.real_env_kuavo import KuavoEnv, FakeRobot
from diffusion_policy.real_world.real_env_SongLing import SongLingEnv, ObsBuffer
# from diffusion_policy.real_world.spacemouse_shared_memory import Spacemouse
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.real_inference_util import (
    get_real_obs_resolution, 
    get_real_obs_dict)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.cv2_util import get_image_transform
import rospy
# import rosbag
input="/relative_folder/SATA/latest.ckpt"
output="/relative_folder/SATA/output"
vis_camera_idx = 1  # camera_f
rosbag_path = "/relative_folder/SATA/dataset/pick_place_241021_kcds21f/pick_place_2024-10-21-21-07-12.bag"

input="/app/action_state/latest.ckpt"
output="/app/action_state/output"
robot_ip="192.168.0.204"
match_dataset="/app/data/SongLing/SongLingPickPlace.zarr"

match_episode=None
init_joints=False
steps_per_inference=15
max_duration=60
frequency=10
command_latency=0.01
OmegaConf.register_new_resolver("eval", eval, replace=True)

# @click.command()
# @click.option('--input', '-i', required=True, help='Path to checkpoint')
# @click.option('--output', '-o', required=True, help='Directory to save recording')
# @click.option('--robot_ip', '-ri', required=True, help="UR5's IP address e.g. 192.168.0.204")
# @click.option('--match_dataset', '-m', default=None, help='Dataset used to overlay and adjust initial condition')
# @click.option('--match_episode', '-me', default=None, type=int, help='Match specific episode from the match dataset')
# @click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
# @click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
# @click.option('--steps_per_inference', '-si', default=6, type=int, help="Action horizon for inference.")
# @click.option('--max_duration', '-md', default=60, help='Max duration for each epoch in seconds.')
# @click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
# @click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SapceMouse command to executing on Robot in Sec.")


def main():
    
    # bag = rosbag.Bag(rosbag_path, 'r')
    # for topic, msg, t in bag.read_messages(topics=[ 
    #         '/camera_f/color/image_raw',\
    #         '/camera_r/color/image_raw',
    #       ]):
    #     if topic=='/camera_f/color/image_raw':
    #         np_arr = np.frombuffer(msg.data, np.uint8)
    #         try:
    #             cv_img = np_arr.reshape((480, 640, 3))  # 这里根据实际图像尺寸调整
    #             cv_img = cv2.resize(cv_img, (256, 256))
    #             # always show the image
    #             cv2.imshow('default_f', cv_img[..., ::-1])
    #             cv2.waitKey(1)
    #         except ValueError as e:
    #             print(f"Error reshaping the image: {e}")
    #     if topic=='/camera_r/color/image_raw':
    #         np_arr = np.frombuffer(msg.data, np.uint8)
    #         try:
    #             cv_img = np_arr.reshape((480, 640, 3))  # 这里根据实际图像尺寸调整
    #             cv_img = cv2.resize(cv_img, (256, 256))
    #             # always show the image which is rgb format
    #             cv2.imshow('default_r', cv_img[..., ::-1])
    #             cv2.waitKey(1)
    #         except ValueError as e:
    #             print(f"Error reshaping the image: {e}")
    #     break
    # bag.close()
    
    steps_per_inference=6
    # load checkpoint
    ckpt_path = input
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # hacks for method-specific setup.
    action_offset = 0
    delta_action = False
    if 'diffusion' in cfg.name:
        # diffusion model
        policy: BaseImagePolicy
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        device = torch.device('cuda')
        policy.eval().to(device)

        # set inference params
        policy.num_inference_steps = 16 # DDIM inference iterations
        policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1

    elif 'robomimic' in cfg.name:
        # BCRNN model
        policy: BaseImagePolicy
        policy = workspace.model

        device = torch.device('cuda')
        policy.eval().to(device)

        # BCRNN always has action horizon of 1
        steps_per_inference = 1
        action_offset = cfg.n_latency_steps
        delta_action = cfg.task.dataset.get('delta_action', False)

    elif 'ibc' in cfg.name:
        policy: BaseImagePolicy
        policy = workspace.model
        policy.pred_n_iter = 5
        policy.pred_n_samples = 4096

        device = torch.device('cuda')
        policy.eval().to(device)
        steps_per_inference = 1
        action_offset = 1
        delta_action = cfg.task.dataset.get('delta_action', False)
    else:
        raise RuntimeError("Unsupported policy type: ", cfg.name)

    # setup experiment
    dt = 1/frequency

    obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    n_obs_steps = cfg.n_obs_steps
    print("n_obs_steps: ", n_obs_steps)
    print("steps_per_inference:", steps_per_inference)
    print("action_offset:", action_offset)

    action_dim = 7
    rospy.init_node("test")
    with SongLingEnv(
        frequency=frequency,
        n_obs_steps=2,
        video_capture_fps=30,
        robot_publish_rate=200,
        
        img_buffer_size=30,
        robot_state_buffer_size=200,
        
        video_capture_resolution=(640, 480),
        output_dir=output,
        ) as env:
            print("waiting for the obs buffer to be ready ......")
            env.obs_buffer.wait_buffer_ready()
            
            '''
            {
                'camera_0': (2, 3, 256, 256),
                'camera_1': (2, 3, 256, 256),
                'state': (2, 7),
                'timestamps': (2, ),
            }
            '''
            print("Warming up policy inference")
            obs, camera_obs, camera_obs_timestamps, robot_obs, robot_obs_timestamps = env.get_obs()
            with torch.no_grad():
                policy.reset()
                obs_dict_np = get_real_obs_dict(
                    env_obs=obs, shape_meta=cfg.task.shape_meta)
                obs_dict = dict_apply(obs_dict_np, 
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                result = policy.predict_action(obs_dict)
                action = result['action'][0].detach().to('cpu').numpy()
                # assert action.shape[-1] == 2
                del result

            output_file = 'SongLing.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4v 编码器
            fps = 10  # 帧率
            vis_img = obs[f'img0{vis_camera_idx}'][-1]
            height, width = vis_img.shape[:2]
            out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
            
            print('Ready!')
            while True:
                # ========= human control loop ==========
                print("skip Human in control!")
                
                # ========== policy control loop ==============
                try:
                    # start episode
                    policy.reset()
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay
                    # env.start_episode(eval_t_start)
                    # wait for 1/30 sec to get the closest frame actually
                    # reduces overall latency
                    frame_latency = 1/30
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    print("Started!")
                    iter_idx = 0
                    term_area_start_timestamp = float('inf')
                    perv_target_pose = None
                    while True:
                        # calculate timing
                        t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

                        # get obs
                        print('get_obs')
                        obs,_,_,_,_ = env.get_obs()
                        obs_timestamps = obs['timestamp']
                        print(f'Obs latency {time.time() - obs_timestamps[-1]}')

                        # run inference
                        with torch.no_grad():
                            s = time.time()
                            obs_dict_np = get_real_obs_dict(
                                env_obs=obs, shape_meta=cfg.task.shape_meta)
                            obs_dict = dict_apply(obs_dict_np, 
                                lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                            result = policy.predict_action(obs_dict)
                            # this action starts from the first obs step
                            action = result['action'][0].detach().to('cpu').numpy()
                            print('Inference latency:', time.time() - s)    # 0.4s
                        
                        # # clip actions
                        # this_target_poses[:,:2] = np.clip(
                        #     this_target_poses[:,:2], [0.25, -0.45], [0.77, 0.40])

                        # execute actions
                        env.exec_actions(
                            actions=action[:],
                        )
                        print(f"Submitted {len(action)} steps of actions.")

                        # visualize
                        # episode_id = env.replay_buffer.n_episodes
                        vis_img = obs[f'img0{vis_camera_idx}'][-2:]
                        # text = 'Episode: {}, Time: {:.1f}'.format(
                        #     episode_id, time.monotonic() - t_start
                        # )
                        
                        for i in range(len(vis_img)):
                            text = 'SongLing Task: Time: {:.1f}'.format(time.monotonic() - t_start)
                            cv2.putText(
                            vis_img[i],
                            text,
                            (10,20),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            thickness=1,
                            color=(255,255,255)
                            )
                            
                            # cv2.imshow('default', vis_img[i][...,::-1])
                            out.write(vis_img[i][..., ::-1])  # 写入视频，转换为 RGB 格式
                            cv2.imwrite(f'default{i}.jpg', vis_img[i][...,::-1])  # 写入图片
                        # write in a mp4 video
                        

                        key_stroke = cv2.pollKey()
                        if key_stroke == ord('s'):
                            # Stop episode
                            # Hand control back to human
                            # env.end_episode()
                            
                            # shut down the out video
                            out.release()
                            print('Stopped.')
                            break

                        # # auto termination
                        # terminate = False
                        # if time.monotonic() - t_start > max_duration:
                        #     terminate = True
                        #     print('Terminated by the timeout!')

                        # if terminate:
                        #     # env.end_episode()
                        #     break

                        # wait for execution
                        # precise_wait(t_cycle_end - frame_latency)
                        # iter_idx += steps_per_inference

                except KeyboardInterrupt:
                    print("Interrupted!")
                    # stop robot.
                    # env.end_episode()
                    env.close()
                    exit(0)
                print("Stopped.")



# %%
if __name__ == '__main__':
    main()
