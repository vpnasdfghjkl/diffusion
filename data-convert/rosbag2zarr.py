import rosbag
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from PIL import Image
import glob
import time
from tqdm import tqdm
from typing import List, Tuple, Dict
from collections import defaultdict
import argparse
import shutil

from config import Config
from replay_buffer import ReplayBuffer
# json_file = "/home/camille/IL/Company/kuavodatalab/data-train-deploy/src/config/Task2-RearangeToy.json"


class RosbagReader:
    def __init__(self, bag_name: str,config: Config, save_plt_folder: str, save_lastPic_folder: str):
        self.bag_name = bag_name
        self.save_plt_folder = save_plt_folder
        self.save_lastPic_folder = save_lastPic_folder
        self.base_name = os.path.splitext(os.path.basename(bag_name))[0]
        self.config = config
        self.video_frames = {}


    def _get_processor(self, topic):
        # 从 config 中获取对应的话题处理函数
        if topic in config.TOPIC_PROCESSOR_MAP:
            return config.TOPIC_PROCESSOR_MAP[topic]
        else:
            raise ValueError(f"No processor defined for topic: {topic}")
        

        
    def _collect_bag_data(self) -> Dict:
        """
        Read and collect all necessary data from rosbag
        args:
        return: Dict
        """
        data = defaultdict(list)

        topics = {
            **{v['topic']: k for k, v in config.DEFAULT_OBS_KEY_MAP['img'].items()},
            **{v['topic']: k for k, v in config.DEFAULT_OBS_KEY_MAP['low_dim'].items()}
        }

        with rosbag.Bag(self.bag_name, 'r') as bag:
            for topic, msg, t in bag.read_messages(topics=list(topics.keys())):
                key = topics[topic]
                processor = self._get_processor(topic)
                ret_dict = processor(msg)
                data[f"{key}_time_stamp"].append(ret_dict["timestamp"])
                data[key].append(ret_dict["data"])
                
        for img in ['img01', 'img02']:
            self.video_frames[img] = data[img][self.config.SAMPLE_DROP:-self.config.SAMPLE_DROP]
            os.makedirs(raw_video_folder + f"/{self.base_name}", exist_ok=True)
            self._save_video(output_video=raw_video_folder + f"/{self.base_name}/{img}.mp4", img_array=self.video_frames[img])
        return data

    def _find_nearest_index(self, time_stamps, target_time) -> int:
        """Find the index of the nearest timestamp to target_time"""
        time_array = np.array([t for t in time_stamps])
        return np.argmin(np.abs(time_array - target_time))
    
    def _integrate_hand_data(self, data: np.ndarray, hand_data: np.ndarray) -> np.ndarray:
        """
        Integrate hand data into data array at specified positions based on array length
        Args:
            data: Joint or EEF pose data
            hand_data: Hand state data
        Returns:
            np.ndarray: Integrated data array
        """
        result = []
        for arr, hand in zip(data, hand_data):
            # Convert to list for easier insertion
            arr_list = list(arr)
            hand = list(hand)
            half_len = len(arr_list) // 2
            half_hand_len = len(hand) // 2
            left_dex = arr_list[:half_len] + hand[:half_hand_len]
            right_dex = arr_list[half_len:] + hand[half_hand_len:]
                
            result.append(left_dex + right_dex)
        return np.array(result, dtype=np.float32)
    
    def _align_data(self, data: Dict, config: Config) -> Dict:
        """Align all data based on image timestamps"""
        jump = config.CAM_HZ // config.TRAIN_HZ
        img_stamps = data['img01_time_stamp'][config.SAMPLE_DROP:-config.SAMPLE_DROP][::jump]
        imgs = np.array(data['img01'][config.SAMPLE_DROP:-config.SAMPLE_DROP][::jump])
        
        aligned_data = defaultdict(list)
        
        # Align all data types to image timestamps
        for stamp in img_stamps:
            stamp_sec = stamp
            for key in config.DEFAULT_TOPIC_2_NAME.values():
                idx = self._find_nearest_index(data[f"{key}_time_stamp"], stamp_sec)
                aligned_data[key].append(data[key][idx])
            aligned_data['timestamp'].append(stamp_sec)

        # Convert lists to numpy arrays and apply sampling
        for key in aligned_data:
            if key in ['img01', 'img02', 'img03']:
                aligned_data[key] = np.array(aligned_data[key], dtype=np.uint8)
            else:
                aligned_data[key] = np.array(aligned_data[key])
            
        # Integrate hand data
        aligned_data['cmd_joint_with_hand'] = self._integrate_hand_data(
            aligned_data['cmd_joint'], 
            aligned_data['cmd_hand'],
        )
        aligned_data['state_joint_with_hand'] = self._integrate_hand_data(
            aligned_data['state_joint'], 
            aligned_data['state_hand'],
        )
        aligned_data['cmd_eef_with_hand'] = self._integrate_hand_data(
            aligned_data['cmd_eef'], 
            aligned_data['cmd_hand'],
        )
        aligned_data['state_eef_with_hand'] = self._integrate_hand_data(
            aligned_data['state_eef'], 
            aligned_data['state_hand'],
        )

        # Calculate delta command poses
        cmd_eef = aligned_data['cmd_eef_with_hand']
        half_len = len(cmd_eef[0]) // 2
        
        half_hand_len = len(aligned_data['cmd_hand'][0]) // 2
        delta_cmd_pose_with_hand = np.zeros_like(cmd_eef[1:])
        
        # Update continuous values
        delta_cmd_pose_with_hand = cmd_eef[1:] - cmd_eef[:-1]
        
        
        # Keep discrete values (hand states) unchanged
        delta_cmd_pose_with_hand[:, half_len-half_hand_len:half_len] = cmd_eef[1:, half_len-half_hand_len:half_len]   
        delta_cmd_pose_with_hand[:, 2* half_len - half_hand_len:] = cmd_eef[1:, 2* half_len - half_hand_len:] 

        
            
        # Remove first frame from all data
        result_data = {
            'img01': imgs[1:],
            'img02': aligned_data['img02'][1:],
            'img03': aligned_data['img03'][1:],
            'state_joint_with_hand': aligned_data['state_joint_with_hand'][1:],
            'cmd_joint_with_hand': aligned_data['cmd_joint_with_hand'][1:],
            'state_eef_with_hand': aligned_data['state_eef_with_hand'][1:],
            'cmd_eef_with_hand': aligned_data['cmd_eef_with_hand'][1:],
            'delta_cmd_eef_pose_with_hand': delta_cmd_pose_with_hand,
    
            'state_joint': aligned_data['state_joint'][1:],
            'cmd_joint': aligned_data['cmd_joint'][1:],
            'state_eef': aligned_data['state_eef'][1:],
            'cmd_eef': aligned_data['cmd_eef'][1:],
            'state_hand': aligned_data['state_hand'][1:],
            'cmd_hand': aligned_data['cmd_hand'][1:],
            'timestamp': aligned_data['timestamp'][1:],
        }
        for img in ['img01', 'img02', 'img03']:
            os.makedirs(zarr_video_folder + f"/{self.base_name}", exist_ok=True)
            self._save_video(output_video=zarr_video_folder + f"/{self.base_name}/{img}.mp4", img_array=result_data[img], fps=10)
            os.makedirs(sample_video_folder + f"/{self.base_name}", exist_ok=True)
            self._save_video(output_video=sample_video_folder + f"/{self.base_name}/{img}.mp4", img_array=result_data[img], fps=10)
        # Debug print for shapes
        print("Data shapes after alignment:")
        for key, value in result_data.items():
            print(f"{key}: {value.shape}")
        
        return result_data

    def _plot_results(self, result_data: Dict):
        """Plot and save comparison graphs"""
        num_plots = min(len(result_data['state_joint'][0]), 
                       len(result_data['state_eef'][0]), 16)
        
        fig, axs = plt.subplots(3, 6, figsize=(48, 48))
        fig.suptitle(self.base_name, fontsize=16)
        
        images01 = self.video_frames['img01']
        images02 = self.video_frames['img02']
        img_strip = np.concatenate(np.array(images01[::20]), axis=1)  # Row for images
        img_strip02 = np.concatenate(np.array(images02[::20]), axis=1)  # Row for images02
        
        # same shape
        if img_strip.shape[0] != img_strip02.shape[0]:
            # resize:
            img_strip02 = cv2.resize(img_strip02, (img_strip.shape[1], img_strip.shape[0]))
            
        
        # Stack image strips vertically
        img_strip_combined = np.vstack([img_strip, img_strip02])
        # Set up plt figure
        ACTION_DIM_LABELS = ['x', 'y', 'z', 'rx', 'ry', 'rz', 'a_hand', 'extra']
        JOINT_DIM_LABELS = ['j1', 'j2', 'j3', 'j4', 'j5', 'j6', 'j7', 'j_hand']
        figure_layout = [
            ['image'] * len(JOINT_DIM_LABELS),
            ACTION_DIM_LABELS[:],
            JOINT_DIM_LABELS[:],
        ]
        plt.rcParams.update({'font.size': 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 12])  # Adjust height for two image strips
        fig.suptitle(self.base_name, fontsize=16)
        
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS[:7]):
            axs[action_label].plot(result_data['cmd_eef_with_hand'][:, action_dim], label='cmd_eef_with_hand', alpha=0.5, zorder=1)
            axs[action_label].plot(result_data['state_eef_with_hand'][:, action_dim], label='state_eef_with_hand', alpha=0.5, zorder=1)
            axs[action_label].plot(result_data['delta_cmd_eef_pose_with_hand'][:, action_dim], label='delta_cmd_eef_pose_with_hand', alpha=0.5, zorder=1)
            axs[action_label].set_title(f"motor {action_dim+1} state")
            axs[action_label].set_xlabel('Time in one episode')
            axs[action_label].legend()
            
        for joint_dim, joint_label in enumerate(JOINT_DIM_LABELS):
            axs[joint_label].plot(result_data['cmd_joint_with_hand'][:, joint_dim], label='cmd_joint_with_hand', alpha=0.5, zorder=1)
            axs[joint_label].plot(result_data['state_joint_with_hand'][:, joint_dim], label='state_joint_with_hand', alpha=0.5, zorder=1)
            axs[joint_label].set_title(f"joint {joint_dim+1} state")
            axs[joint_label].set_xlabel('Time in one episode')
            axs[joint_label].legend()
            
        axs['image'].imshow(img_strip_combined)
        axs['image'].set_xlabel('Time in one episode (subsampled)')
        axs['image'].set_title('Image Comparison (Top: images, Bottom: images02)')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"{self.save_plt_folder}/{self.base_name}.png")
        
        img = Image.fromarray(result_data['img01'][-1], 'RGB')
        img.save(f"{self.save_lastPic_folder}/{self.base_name}.png")
        
    def _save_video(self, output_video, img_array, format='mp4', fps=30, img_size=(384, 384)):
        output_video = output_video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        video_writer = cv2.VideoWriter(output_video, fourcc, fps, img_size)

        for img in img_array:
            bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            video_writer.write(bgr_img)
        video_writer.release()
        
    def process_bag(self, config: Config) -> Tuple:
        """Main processing function"""
        # Collect data from bag
        raw_data = self._collect_bag_data()
        
        # Validate data
        if not raw_data['cmd_joint'] or not raw_data['state_joint']:
            raise ValueError("ROS bag file contains empty data for at least one topic.")
        if len(raw_data['cmd_joint']) < 100 or len(raw_data['state_joint']) < 100:
            raise ValueError("ROS bag file data count is too small (less than 100 data points).")
        
        # Process and align data
        result_data = self._align_data(raw_data, config)
        
        # Plot results
        self._plot_results(result_data)
        
        return (result_data['img01'], 
                result_data['img02'], 
                
                result_data['state_joint_with_hand'], 
                result_data['cmd_joint_with_hand'],
                result_data['state_eef_with_hand'], 
                result_data['cmd_eef_with_hand'],
                result_data['state_eef'], 
                result_data['cmd_eef'], 
                result_data['state_joint'], 
                result_data['cmd_joint'],
                result_data['state_hand'], 
                result_data['cmd_hand'],
                
                result_data['delta_cmd_eef_pose_with_hand'],
                result_data['timestamp']
                )

def use_rosbag_to_show(bag_path: str) -> Tuple:
    """Main entry point for processing rosbag data"""
    reader = RosbagReader(bag_path,
                         config=config,
                         save_plt_folder=save_plt_folder,
                         save_lastPic_folder=save_lastPic_folder)
    return reader.process_bag(config)


def process_bag_files(bag_folder_path: str):
    
    task_name = os.path.basename(os.path.dirname(bag_folder_path))

    # Find .bag files
    bag_paths = glob.glob(f"{bag_folder_path}/*.bag")
    print(f"Found {len(bag_paths)} bag files.")
    
    # Define output zarr path
    output_zarr_path = os.path.join(save_zarr_folder, f"{task_name}.zarr")
    
    # Initialize replay buffer
    replay_buffer = ReplayBuffer.create_from_path(output_zarr_path, mode='a')
    
    # Process each bag file
    for path in tqdm(bag_paths, desc="Processing bags", unit="bag"):
        start_time = time.time()
        print(f"Processing {path}")
        
        # Get the seed (number of episodes) from replay buffer
        seed = replay_buffer.n_episodes
        
        # Extract data from rosbag
        # img01, aligned_img02, eef_s, delta_eef_a, eef_a, joint_s, joint_a = use_rosbag_to_show(path)
        img01, img02, state_joint_with_hand, cmd_joint_with_hand, state_eef_with_hand, cmd_eef_with_hand, \
        state_eef,cmd_eef, state_joint, cmd_joint, state_hand, cmd_hand, delta_cmd_eef_pose_with_hand, \
            timestamp = use_rosbag_to_show(path) 
        
        data = list(zip(img01, img02, state_joint_with_hand, cmd_joint_with_hand, state_eef_with_hand, cmd_eef_with_hand, \
        state_eef,cmd_eef, state_joint, cmd_joint, state_hand, cmd_hand, delta_cmd_eef_pose_with_hand,\
            timestamp))
        
        # Create episode list
        episode = [
            {key: val for key, val in zip(
                ['img01', 'img02', 'state_joint_with_hand', 'cmd_joint_with_hand', 'state_eef_with_hand', 'cmd_eef_with_hand', \
                    'state_eef', 'cmd_eef', 'state_joint', 'cmd_joint', 'state_hand', 'cmd_hand', 'delta_cmd_eef_pose_with_hand',\
                        'timestamp'],
                item
            )}
            for item in data
        ]
        
        print(f"Episode length: {len(episode)}")
        
        # Stack data for replay buffer
        data_dict = {key: np.stack([x[key] for x in episode]) for key in episode[0].keys()}
        
        # Add episode to replay buffer
        replay_buffer.add_episode(data_dict, compressors='disk')
        print(f"Saved seed {seed}")
        
        elapsed_time = time.time() - start_time
        print(f"Time taken for {path}: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process rosbag files and save to zarr format")
    
    # Define argument for bag folder path with a default value
    parser.add_argument(
        "--bag_folder_path", 
        type=str, 
        nargs="?",  # Make this argument optional
        default="/home/camille/IL/Company/kuavodatalab/data-convert/data-example/Task1-RearangeToy/kuavo-rosbag",  # Set the default value
        help="The rosbag folder under a task folder, e.g. '/app/data-convert/data-example/Task1-RearangeToy/kuavo-rosbag'"
    )
    # config = Config.from_json("/home/camille/IL/Company/kuavodatalab/data-train-deploy/src/config/Task2-RearangeToy.json")
    parser.add_argument(
        "--config", 
        type=str, 
        default="/home/camille/IL/Company/kuavodatalab/data-train-deploy/src/config/Task2-RearangeToy.json",  # Set the default value
        help="The configuration file path, e.g. '/app/data-train-deploy/src/config/Task2-RearangeToy.json'"
    )
    
    # Parse arguments
    args = parser.parse_args()
    bag_folder_path = args.bag_folder_path  # Corrected variable name
    config_path = args.config
    
    config = Config.from_json(config_path)
    save_folder_base = os.path.join(bag_folder_path, "../plt-check")
    save_plt_folder = os.path.join(save_folder_base, "motor-plt")
    save_lastPic_folder = os.path.join(save_folder_base, "last-pic")
    
    save_zarr_folder = os.path.join(bag_folder_path, "../kuavo-zarr")
    zarr_video_folder = os.path.join(save_zarr_folder, "zarr-video")
    raw_video_folder = os.path.join(bag_folder_path, "../raw-video")
    sample_video_folder = os.path.join(bag_folder_path, "../sample-video")
    
    for folder in [save_plt_folder, save_lastPic_folder, save_zarr_folder, raw_video_folder, sample_video_folder, zarr_video_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)
        else:
            shutil.rmtree(folder)
            os.makedirs(folder)
    process_bag_files(bag_folder_path)


