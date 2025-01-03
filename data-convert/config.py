import json
from dataclasses import dataclass, field
from typing import Dict
from msg_process import ProcessUtil

@dataclass
class Config:
    """Configuration parameters for data processing"""
    TASK_TIME: float
    CAM_HZ: int
    TRAIN_HZ: int
    SAMPLE_DROP: int
    DEFAULT_OBS_KEY_MAP: Dict[str, Dict[str, Dict[str, str]]]
    DEFAULT_TOPIC_2_NAME: Dict[str, str] = field(init=False)
    HAND_OPEN_STATE: str
    HAND_CLOSE_STATE: str
    img_resize:Dict[str, list[int, int]]
    def __post_init__(self):
        """在类初始化后，动态生成 DEFAULT_TOPIC_2_NAME"""
        self.DEFAULT_TOPIC_2_NAME = {
            **{v['topic']: k for k, v in self.DEFAULT_OBS_KEY_MAP['img'].items()},
            **{v['topic']: k for k, v in self.DEFAULT_OBS_KEY_MAP['low_dim'].items()},
        }
        self.TOPIC_PROCESSOR_MAP = {
                "/kuavo_arm_traj": lambda msg: ProcessUtil.process_joint(msg, is_cmd=True),
                "/robot_arm_q_v_tau": lambda msg: ProcessUtil.process_joint(msg, is_cmd=False),
                "/fk/cmd_eef": lambda msg: ProcessUtil.process_record_arm_hand_pose(msg),
                "/fk/state_eef": lambda msg: ProcessUtil.process_record_arm_hand_pose(msg),
                "/robot_hand_eff": lambda msg: ProcessUtil.process_hand_data(msg, is_cmd=True, is_binary=True, HAND_CLOSE_STATE=self.HAND_CLOSE_STATE),
                "/robot_hand_position": lambda msg: ProcessUtil.process_hand_data(msg, is_cmd=False, is_binary=True),
                "/cam_1/color/image_raw/compressed": lambda msg: ProcessUtil.process_compressed_image(msg, resize=(self.img_resize["img01"][0], self.img_resize["img01"][1])),
                "/cam_2/color/image_raw/compressed": lambda msg: ProcessUtil.process_compressed_image(msg, resize=(self.img_resize["img02"][0], self.img_resize["img02"][1])),
            }
    @classmethod
    def from_json(cls, json_file: str):
        """从 JSON 文件加载配置并创建 Config 实例"""
        with open(json_file, 'r') as file:
            config_data = json.load(file)
        return cls(**config_data)

