import rosbag
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
# from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
import shutil

from scipy.spatial.transform import Rotation as R
import glob
import math

# diffusion lib======================================================
from diffusion_policy.common.replay_buffer import ReplayBuffer
import time
from tqdm import tqdm
# diffusion======================================================
# bridge = CvBridge()

# 创建一个 VideoWriter 对象
# 参数依次为：输出文件名、编码方式、帧率、图像大小
output_video = 'output_video.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 或者使用其他编码方式如 'MJPG', 'MP4V', 'X264'
frame_rate = 30  # 可以根据需要调整帧率
img_size = (640, 480)  # 根据你的图像大小调整，例如 cv_img.shape[1], cv_img.shape[0]
video_writer = cv2.VideoWriter(output_video, fourcc, frame_rate, img_size)
CAM_HZ=30
TRAIN_HZ=10
TASK_TIME=1000

def check_folder(CHECK_PIC_SAVE_FOLDER):
    if not os.path.exists(CHECK_PIC_SAVE_FOLDER):
        os.makedirs(CHECK_PIC_SAVE_FOLDER)
    else:
        # 清空文件夹中的所有内容
        shutil.rmtree(CHECK_PIC_SAVE_FOLDER)
        os.makedirs(CHECK_PIC_SAVE_FOLDER)

def adjust_pose_rpy(lr_pose):
    threshold= math.pi
    left_pose = []
    lr_pose = np.array(lr_pose)
    pose = lr_pose[:,:6]
    pre_eef = pose[0][3:6]  
    for i in range(len(pose)):
        for j in range(3): 
            diff = pose[i][3+j] - pre_eef[j]
            if diff > threshold:
                pose[i][3+j] -= 2 * math.pi
            elif diff < -threshold:
                pose[i][3+j] += 2 * math.pi
            pre_eef[j] = pose[i][3+j]
    left_pose.append(pose)  #(1, 1976, 6)

    right_pose = []
    pose = lr_pose[:,6:]
    pre_eef = pose[0][3:6]  
    for i in range(len(pose)):
        for j in range(3): 
            diff = pose[i][3+j] - pre_eef[j]
            if diff > threshold:
                pose[i][3+j] -= 2 * math.pi
            elif diff < -threshold:
                pose[i][3+j] += 2 * math.pi
            pre_eef[j] = pose[i][3+j]
    right_pose.append(pose) #(1, 1976, 6)

    # (1976,12)
    all_pose =np.concatenate((left_pose[0],right_pose[0]),axis=1)
    return np.array(all_pose)

def plot_euler_error(cmd_rot_matrix,state_rot_matrix,base_name):
    euler_error = []

    for (c_m,s_m) in zip(cmd_rot_matrix,state_rot_matrix):
        delta_rot_matrix = np.dot(c_m, s_m.T)
        euler_error.append(R.from_matrix(delta_rot_matrix).as_euler('xyz'))
    # plot euler_error
    euler_error = np.array(euler_error[300:])
    fig, axs = plt.subplots(3, 1, figsize=(32, 18))
    fig.suptitle(base_name, fontsize=16)
    for i in range(3):
        axs[i].plot(euler_error[:, i], label=f'error_{i}')
        axs[i].set_title(f"error_{i}")
        axs[i].legend()
    plt.tight_layout()
    save_path = f"{save_plt_folder}/{base_name}_euler_error.png"
    plt.savefig(save_path)
    
def plot_euler_error_direct(cmd_eef_pose,state_eef_pose,base_name):
    # plot euler_error_direct
    euler_error_direct=[]
    for (c_e,s_e) in zip(cmd_eef_pose,state_eef_pose):
        delta_euler = c_e[3:6] - s_e[3:6]
        euler_error_direct.append(delta_euler)
    euler_error_direct = np.array(euler_error_direct[300:])
    fig, axs = plt.subplots(3, 1, figsize=(32, 18))
    fig.suptitle(base_name, fontsize=16)
    for i in range(3):
        axs[i].plot(euler_error_direct[:, i], label=f'error_direct_{i}')
        axs[i].set_title(f"error_direct_{i}")
        axs[i].legend()
    plt.tight_layout()
    save_path = f"{save_plt_folder}/{base_name}_euler_error_direct.png"
    plt.savefig(save_path)
    
def use_rosbag_to_show(bag_name):
    base_name = os.path.splitext(os.path.basename(bag_name))[0]
    # 读取rosbag文件并提取所需数据
    bag = rosbag.Bag(bag_name, 'r')

    start_time = bag.get_start_time()
    end_time = start_time + TASK_TIME

    cmd_joint=[]
    cmd_joint_time_stamp=[]
    state_joint=[]
    state_joint_time_stamp=[]

    cmd_eef_pose=[]
    cmd_eef_pose_time_stamp=[]
    state_eef_pose=[]
    state_eef_pose_time_stamp=[]

    cmd_hand=[]
    cmd_hand_time_stamp=[]
    state_hand=[]
    state_hand_time_stamp=[]

    img=[]
    img_stamp=[]
    img02=[]
    img02_stamp=[]

    cmd_rot_matrix = []
    state_rot_matrix = []
    delta_rot_matrix =[]
    
    for topic, msg, t in bag.read_messages(topics=[ '/kuavo_arm_traj',\
                                                    '/robot_arm_q_v_tau',\
                                                    '/drake_ik/cmd_arm_hand_pose',\
                                                    '/drake_ik/real_arm_hand_pose', \
                                                    '/robot_hand_eff',\
                                                    '/robot_hand_position',\
                                                    '/camera1/color/image_raw/compressed',\
                                                    '/camera2/color/image_raw/compressed',
                                                  ]):
        # msg_time = msg.header.stamp.to_sec()  # 将时间戳转换为秒
        # if msg_time > end_time:
        #     break  # 超过时间限制，停止读取
        
        if topic == '/kuavo_arm_traj':
            cmd_joint_time_stamp.append(msg.header.stamp)
            cmd_joint.append(np.radians(msg.position)[:])

        elif topic == '/robot_arm_q_v_tau':
            state_joint_time_stamp.append(msg.header.stamp)
            state_joint.append((msg.q)[:])
            
        elif topic=='/drake_ik/cmd_arm_hand_pose':
            cmd_eef_pose_time_stamp.append(msg.header.stamp)
            left_right_eef = [
                np.concatenate((np.array(pose.pos_xyz), R.from_quat(pose.quat_xyzw).as_euler('xyz')))
                for pose in [msg.left_pose, msg.right_pose]
            ]
            cmd_eef_pose.append(np.concatenate(left_right_eef))
            
            
        elif topic=='/drake_ik/real_arm_hand_pose':
            state_eef_pose_time_stamp.append(msg.header.stamp)
            left_right_eef = [
                np.concatenate((np.array(eef.pos_xyz), R.from_quat(eef.quat_xyzw).as_euler('xyz')))
                for eef in [msg.left_pose, msg.right_pose]
            ]
            state_eef_pose.append(np.concatenate(left_right_eef))

        elif topic=='/robot_hand_eff':
            cmd_hand_time_stamp.append(msg.header.stamp)
            dex_hand=msg.data
            l_bi_grip = 0 if dex_hand[5] == 0 else 1
            r_bi_grip = 0 if dex_hand[11] == 0 else 1
            bi_grip = [l_bi_grip] + [r_bi_grip]
            cmd_hand.append(bi_grip)

        elif topic=='/robot_hand_position':
            state_hand_time_stamp.append(msg.header.stamp)
            dex_hand= [float(i) for i in msg.left_hand_position + msg.right_hand_position]
            l_bi_grip = 0 if dex_hand[5] == 0 else 1
            r_bi_grip = 0 if dex_hand[11] == 0 else 1
            bi_grip = [l_bi_grip] + [r_bi_grip]
            state_hand.append(bi_grip)

        elif topic=='/camera1/color/image_raw/compressed':
            img_stamp.append(msg.header.stamp)
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # 使用 cv2 解码图像
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            # cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            resized_img = cv2.resize(cv_img, (256, 256))
            img.append(resized_img)

        elif topic=='/camera2/color/image_raw/compressed':
            img02_stamp.append(msg.header.stamp)
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # 使用 cv2 解码图像
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            # cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            resized_img = cv2.resize(cv_img, (256, 256))
            img02.append(resized_img)
           
    bag.close()
    cmd_eef_pose=adjust_pose_rpy(cmd_eef_pose)
    state_eef_pose=adjust_pose_rpy(state_eef_pose)

    # plot_euler_error(cmd_rot_matrix,state_rot_matrix,base_name)
    # plot_euler_error_direct(cmd_eef_pose,state_eef_pose,base_name)
    
    # 安全判断
    if len(cmd_joint) == 0 or len(state_joint) == 0:
        print("ROS bag file contains empty data for at least one topic.")
        return

    if len(cmd_joint) < 100 or len(state_joint) < 100:
        print("ROS bag file data count is too small (less than 100 data points). Please check again.")
        return
    

    aligned_state_joint = []
    aligned_cmd_joint = []
    aligned_state_hand=[]
    aligned_cmd_hand=[]
    aligned_cmd_eef_pose=[]
    aligned_state_eef_pose=[]
    aligned_img02=[]

    drop=2
    img=img[drop:-drop]
    img_stamp=img_stamp[drop:-drop]
    assert len(img)==len(img_stamp)
    for stamp in img_stamp:
        stamp_sec=stamp.to_sec()
        idx_s = np.argmin(np.abs(np.array([t.to_sec() for t in state_joint_time_stamp]) - stamp_sec))
        aligned_state_joint.append(state_joint[idx_s])

        idx_a = np.argmin(np.abs(np.array([t.to_sec() for t in cmd_joint_time_stamp]) - stamp_sec))
        aligned_cmd_joint.append(cmd_joint[idx_a])

        idx_h=np.argmin(np.abs(np.array([t.to_sec() for t in state_hand_time_stamp]) - stamp_sec))
        aligned_state_hand.append(state_hand[idx_h])
        
        idx_h=np.argmin(np.abs(np.array([t.to_sec() for t in cmd_hand_time_stamp]) - stamp_sec))
        aligned_cmd_hand.append(cmd_hand[idx_h])

        idx_h=np.argmin(np.abs(np.array([t.to_sec() for t in cmd_eef_pose_time_stamp]) - stamp_sec))
        aligned_cmd_eef_pose.append(cmd_eef_pose[idx_h])

        idx_h=np.argmin(np.abs(np.array([t.to_sec() for t in state_eef_pose_time_stamp]) - stamp_sec))
        aligned_state_eef_pose.append(state_eef_pose[idx_h])

        idx_img2=np.argmin(np.abs(np.array([t.to_sec() for t in img02_stamp]) - stamp_sec))
        aligned_img02.append(img02[idx_img2])

    aligned_cmd_joint = [list(item) for item in aligned_cmd_joint]
    aligned_state_joint = [list(item) for item in aligned_state_joint]
    aligned_cmd_eef_pose=[list(item) for item in aligned_cmd_eef_pose]
    aligned_state_eef_pose=[list(item) for item in aligned_state_eef_pose]
    
    print("all length==============>:\nimg_stamp,aligned_cmd_joint,aligned_state_joint,aligned_cmd_eef_pose,aligned_state_eef_pose,aligned_cmd_hand,aligned_state_hand")
    print(len(img_stamp),len(aligned_cmd_joint),len(aligned_state_joint),len(aligned_cmd_eef_pose),len(aligned_state_eef_pose),len(aligned_cmd_hand),len(aligned_state_hand))
    assert len(img_stamp)==len(aligned_cmd_joint)==len(aligned_state_joint)==len(aligned_cmd_eef_pose)==len(aligned_state_eef_pose)==len(aligned_cmd_hand)==len(aligned_state_hand)
    
    for i in range(len(img_stamp)):
        # insert hand[0] ,hand[1] to joint[6], joint[13]
        aligned_cmd_joint[i].insert(6,aligned_cmd_hand[i][0])
        aligned_cmd_joint[i].insert(13,aligned_cmd_hand[i][1])
        aligned_state_joint[i].insert(6,aligned_state_hand[i][0])
        aligned_state_joint[i].insert(13,aligned_state_hand[i][1])
        aligned_cmd_eef_pose[i].insert(6,aligned_cmd_hand[i][0])
        aligned_cmd_eef_pose[i].insert(13,aligned_cmd_hand[i][1])
        aligned_state_eef_pose[i].insert(6,aligned_state_hand[i][0])
        aligned_state_eef_pose[i].insert(13,aligned_state_hand[i][1])
        # aligned_cmd_joint[i].append(aligned_cmd_hand[i])
        # aligned_state_joint[i].append(aligned_state_hand[i])
        # aligned_cmd_eef_pose[i].append(aligned_cmd_hand[i])
        # aligned_state_eef_pose[i].append(aligned_state_hand[i])

 # s 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
 # a 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15

    jump=CAM_HZ//TRAIN_HZ
    aligned_cmd_joint=np.array(aligned_cmd_joint)[::jump].astype(np.float32)
    aligned_state_joint=np.array(aligned_state_joint)[::jump].astype(np.float32)
    aligned_cmd_eef_pose=np.array(aligned_cmd_eef_pose)[::jump].astype(np.float32)
    aligned_state_eef_pose=np.array(aligned_state_eef_pose)[::jump].astype(np.float32)
    aligned_delta_cmd_eef_pose=None
    img=img[::jump]
    aligned_img02=aligned_img02[::jump]


    print("after jump, all length==============>:")
    print(len(img),len(aligned_img02),len(aligned_state_eef_pose),len(aligned_state_joint),len(aligned_cmd_joint))
    assert len(img)==len(aligned_img02)==len(aligned_state_eef_pose)==len(aligned_cmd_eef_pose)==len(aligned_state_joint)==len(aligned_cmd_joint)

    aligned_cmd_joint=aligned_cmd_joint[1:]
    aligned_state_joint=aligned_state_joint[1:]

    aligned_state_eef_pose=aligned_state_eef_pose[1:]
    aligned_delta_cmd_eef_pose=aligned_cmd_eef_pose[1:]-aligned_cmd_eef_pose[:-1]
    aligned_delta_cmd_eef_pose[:,6]=aligned_cmd_eef_pose[1:,6]
    aligned_delta_cmd_eef_pose[:,13]=aligned_cmd_eef_pose[1:,13]
    aligned_cmd_eef_pose=aligned_cmd_eef_pose[1:]
    img=img[1:]
    aligned_img02=aligned_img02[1:]

    print("after delete firet frame==============>:")
    print(len(img),len(aligned_state_eef_pose),len(aligned_state_joint),len(aligned_cmd_joint),len(aligned_delta_cmd_eef_pose),len(aligned_cmd_eef_pose))


    # import matplotlib
    # matplotlib.use('Agg')
    # 创建3行5列的图表并进行比较
    num_plots = min(len(aligned_state_joint[0]), len(aligned_state_eef_pose[0]), 16)  # 限制最多只显示15个数据对比
    fig, axs = plt.subplots(3, 6, figsize=(64, 36))
    fig.suptitle(base_name, fontsize=16)
    for i in range(num_plots):
        kuavo_position = [data[i] for data in aligned_cmd_joint]
        robot_q = [data[i] for data in aligned_state_joint]

        cmd_eef=[data[i] for data in aligned_cmd_eef_pose]
        state_eef=[data[i] for data in aligned_state_eef_pose]
        cmd_eef_delta=[data[i] for data in aligned_delta_cmd_eef_pose]
        row = i // 6
        col = i % 6
        # axs[row, col].plot(kuavo_position, label='/kuavo_arm_traj')
        # axs[row, col].plot(robot_q, label='/robot_arm_q_v_tau')
        axs[row, col].plot(cmd_eef, label='/cmd_eef')
        axs[row, col].plot(state_eef, label='/state_eef')
        axs[row, col].plot(cmd_eef_delta, label='/cmd_eef_delta')
        axs[row, col].set_title(f"motor {i+1} state")
        axs[row, col].legend()

    exampl_index=50
    print(f"example index {exampl_index}:")
    print(" cmd_joint:",aligned_cmd_joint[exampl_index],
          "\n state_joint:",aligned_state_joint[exampl_index],
          "\n aligned_delta_cmd_eef_pose:",aligned_delta_cmd_eef_pose[exampl_index],
          "\n state_eef:",aligned_state_eef_pose[exampl_index],
          "\n img shape:",img[exampl_index].shape,
          "\n aligned_img02 shape:",aligned_img02[exampl_index].shape)   

    plt.tight_layout()
    
    

    # 保存图片
    save_path = f"{save_plt_folder}/{base_name}.png"
    plt.savefig(save_path)

    # 保存最后一张img
    cv2.imwrite(f"{save_lastPic_folder}/{base_name}.png",img[-1])
    # # 显示图片
    # plt.show()
    assert len(img)==len(aligned_state_eef_pose)==len(aligned_delta_cmd_eef_pose)==len(aligned_cmd_eef_pose)==len(aligned_state_joint)==len(aligned_cmd_joint)
    print("all length==============>:img,aligned_state_eef_pose,aligned_delta_cmd_eef_pose,aligned_cmd_eef_pose,aligned_state_joint,aligned_cmd_joint")
    print(len(img),len(aligned_state_eef_pose),len(aligned_delta_cmd_eef_pose),len(aligned_cmd_eef_pose),len(aligned_state_joint),len(aligned_cmd_joint))   
    
    # output_zarr_path = f"{save_zarr_folder}/{base_name}.zarr"
    
    return img,aligned_img02,aligned_state_eef_pose,aligned_delta_cmd_eef_pose,aligned_cmd_eef_pose,aligned_state_joint,aligned_cmd_joint

if __name__ == "__main__":
    bag_folder_name = "toy_1"
    bag_folder_path = "/home/lab/hanxiao/dataset/kuavo/task_toy/" + bag_folder_name
    
    save_plt_folder = f"{bag_folder_path}/plt"
    save_lastPic_folder = f"{bag_folder_path}/last_pic"
    save_zarr_folder = f"{bag_folder_path}/zarr"  
    
    check_folder(save_plt_folder)
    check_folder(save_lastPic_folder)
    # check_folder(save_zarr_folder)  
    bagpath = glob.glob(f"{bag_folder_path}/*.bag")
    print(bagpath)
    print(len(bagpath))
    output_zarr_path = f"{save_zarr_folder}/{bag_folder_name}.zarr"
    
    replay_buffer = ReplayBuffer.create_from_path(output_zarr_path, mode='a')

    for path in tqdm(bagpath, desc="Processing bags", unit="bag"):
        start_time = time.time()
        print("current path", path)
        seed = replay_buffer.n_episodes
        img01,aligned_img02, eef_s, delta_eef_a, eef_a, joint_s, joint_a = use_rosbag_to_show(path)
        data = list(zip(img01,aligned_img02, eef_s, delta_eef_a, eef_a, joint_s, joint_a))
        
        episode = []
        for i, (img01, img02, eef_s, delta_eef_a, eef_a, joint_s, joint_a,) in enumerate(data):
            episode.append({
                'img01': img01,
                'img02': img02,
                'eef_s': eef_s,
                'delta_eef_a': delta_eef_a,
                'eef_a': eef_a,
                'joint_s': joint_s,
                'joint_a': joint_a,
            })

        print("episode length", len(episode))
        print("episode keys", episode[0].keys())
        print("episode[0]['img'].shape", episode[0]['img01'].shape)
        print("episode[0]['state'].shape", episode[0]['eef_s'].shape)
        print("episode[0]['action'].shape", episode[0]['delta_eef_a'].shape)

        data_dict = dict()
        for key in episode[0].keys():
            data_dict[key] = np.stack([x[key] for x in episode])
        replay_buffer.add_episode(data_dict, compressors='disk')
        print(f'saved seed {seed}')

        elapsed_time = time.time() - start_time
        print(f"Time taken for {path}: {elapsed_time:.2f} seconds")
