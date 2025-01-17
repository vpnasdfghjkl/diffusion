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
# from diffusion_policy.real_world.real_env_kuavo import KuavoEnv
# from diffusion_policy.real_world.real_env_kuavo_cup import KuavoEnv
from diffusion_policy.real_world.real_env_kuavo_B import KuavoEnv
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
import sys, signal
import threading
from queue import Queue
import multiprocessing

input="/app/diffusion/data/outputs/2024.12.01/20.25.02_train_diffusion_unet_image_KuavoToy/checkpoints/latest.ckpt" # joint
input="/home/lab/hanxiao/diffusion/data/outputs/2024.12.01/20.25.02_train_diffusion_unet_image_KuavoToy/checkpoints/latest.ckpt" # joint
input="/home/lab/hanxiao/diffusion/data/outputs/2024.12.28/00.25.07_train_diffusion_unet_image_KuavoToy/checkpoints/epoch=0250-train_loss=0.002.ckpt" # joint
input="/home/lab/hanxiao/diffusion/data/outputs/2025.01.15/15.30.25_train_diffusion_unet_image_KuavoGrabCup/checkpoints/epoch=0050-train_loss=0.011.ckpt"
input="/home/lab/hanxiao/diffusion/data/outputs/2025.01.16/22.24.19_train_diffusion_unet_image_KuavoGrabB/checkpoints/epoch=0100-train_loss=0.012.ckpt"

output="/home/lab/hanxiao/diffusion/data/outputs/2024.12.01/20.25.02_train_diffusion_unet_image_KuavoToy/checkpoints/output" # joint
output="/home/lab/hanxiao/diffusion/data/outputs/2024.11.28/15.13.53_train_diffusion_unet_image_KuavoToy/checkpoints/output"
output="/home/lab/hanxiao/diffusion/data/outputs/2024.12.28/00.25.07_train_diffusion_unet_image_KuavoToy/checkpoints/output" # joint
output="/home/lab/hanxiao/diffusion/data/outputs/2025.01.15/15.30.25_train_diffusion_unet_image_KuavoGrabCup/checkpoints/output"
output="/home/lab/hanxiao/diffusion/data/outputs/2025.01.16/22.24.19_train_diffusion_unet_image_KuavoGrabB/checkpoints/output"

vis_camera_idx=1
steps_per_inference=6
max_duration=60
frequency=10
command_latency=0.01
OmegaConf.register_new_resolver("eval", eval, replace=True)


def main():
    # load match_dataset
    match_camera_idx = 0
    episode_first_frame_map = dict()
    
    # load checkpoint
    steps_per_inference=6
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
    else:
        raise RuntimeError("Unsupported policy type: ", cfg.name)

    # setup experiment
    dt = 1/frequency

    obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    n_obs_steps = cfg.n_obs_steps
    print("n_obs_steps: ", n_obs_steps)
    print("steps_per_inference:", steps_per_inference)
    print("action_offset:", action_offset)

    rospy.init_node("test")
    with KuavoEnv(
        frequency=frequency,
        n_obs_steps=2,
        video_capture_fps=30,
        robot_publish_rate=100,

        img_buffer_size=30,
        robot_state_buffer_size=100,

        video_capture_resolution=(640, 480),

        output_dir=output,
        ) as env:
            print("waiting for the obs buffer to be ready ......")
            time.sleep(1.0)
            env.obs_buffer.wait_buffer_ready(just_img=True)
            '''
            {
                'camera_0': (2, 256, 256, 3,),
                'camera_1': (2, 256, 256, 3,),
                'state': (2, 7),
                'timestamps': (2, ),
            }
            '''
            print("Warming up policy inference")
            latency = 0
            current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            output_video_path = f'./{cfg.task.name}_{current_time}_speedx1_latency{latency}ms.mp4'
            stop_event = threading.Event()  
            record_title = f"Latency of [Obs_trans, PredictRet_trans, Motor] =[{latency}ms, {latency}ms, {latency}ms]"
            video_thread = threading.Thread(target=env.record_video, args=(output_video_path, 640, 480, 30, True,stop_event,record_title))
            video_thread.start()
            
            def handle_exit_signal(signum, frame, stop_event):
                print("Signal received, saving video and cleaning up...")
                stop_event.set()  # 停止视频录制
                cv2.destroyAllWindows()  # 关闭所有OpenCV窗口
                sys.exit(0)  # 退出程序
                
            # 注册信号处理器
            signal.signal(signal.SIGINT, lambda sig, frame: handle_exit_signal(sig, frame, stop_event))
            signal.signal(signal.SIGQUIT, lambda sig, frame: handle_exit_signal(sig, frame, stop_event))
            
            
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
                    
                    
                    import matplotlib.pyplot as plt

                    
                    import imageio
                    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
                    output_video_path = f'./{cfg.task.name}_{current_time}_speedx_latency{latency}ms.mp4'
                    writer = imageio.get_writer(output_video_path, fps=10, codec="libx264")
                    while True:
                        # calculate timing
                        t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

                        # get obs
                        # print('get_obs')
                        obs,_,_,_,_ = env.get_obs()
                        start_point = time.time()
                        time.sleep(latency/1000)
                        obs_timestamps = obs['timestamp']
                        # print(f'Obs latency {time.time() - obs_timestamps[-1]}')
                        

                        imgs = []
                        for k, v in obs.items():
                            if "img" in k:
                                img = obs[k][-1]
                                img = cv2.resize(img, (640, 480))
                                img01_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                                imgs.append(img01_bgr)
                                
                        concatenated_img = np.hstack(imgs)  # 横向拼接图像
                        cv2.imshow('Image Stream', concatenated_img)
                        
                        from PIL import Image, ImageDraw, ImageFont
                        concatenated_img = cv2.cvtColor(concatenated_img, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(concatenated_img)
                        draw = ImageDraw.Draw(pil_img)
                        font = ImageFont.load_default()  # 使用默认字体，可以选择其他字体
                        title = record_title
                        draw.text((50, 50), title, font=font, fill="green")
                        
                        # 转回Numpy数组
                        concatenated_img_with_text = np.array(pil_img)
                        
                        # 写入视频
                        writer.append_data(concatenated_img_with_text)
                        

                        # 处理按键事件，按'q'退出
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                        
                        # writer.append_data(cv2.cvtColor(concatenated_img, cv2.COLOR_BGR2RGB))

                        

                        with open('inference_latency.txt', 'a') as f:
                            # run inference
                            with torch.no_grad():
                                obs_dict_np = get_real_obs_dict(
                                    env_obs=obs, shape_meta=cfg.task.shape_meta)
                                print("eef_s", obs_dict_np['agent_pos'][-1])
                                obs_dict = dict_apply(obs_dict_np, 
                                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                                s = time.time()
                                result = policy.predict_action(obs_dict)
                                inference_time = time.time() - s
                                
                                    
                                print('Inference latency:', inference_time)
                                f.write(f"Inference latency: {inference_time} seconds\n")
                                # this action starts from the first obs step
                                action = result['action'][0].detach().to('cpu').numpy()
                                # 计算推理所用的时长
                                # print('Inference latency:', time.time() - s)
                        time.sleep(latency/1000)
                        # # clip actions
                        # this_target_poses[:,:2] = np.clip(
                        #     this_target_poses[:,:2], [0.25, -0.45], [0.77, 0.40])

                        # execute actions
                        for act in action:
                            print("act", act)
                            break
                        print("\n\n")
                        print('--------------------------',time.time() - start_point)
                        env.exec_actions(
                            actions=action[:8],cur_state=obs_dict_np['agent_pos'][-1][:6],start_point=start_point,latency=latency/1000,
                        )
                        # print(f"Submitted {len(action)} steps of actions.")

                    cv2.destroyAllWindows()

                     

                       
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
