# data and train
## data convert from rosbag to zarr
### 1. init content strucutre:
```
data-convert
├── config.py
├── data-example
│   └── Task2-RearangeToy
│       └── kuavo-rosbag
│           ├── task_pcd_test_2024-12-27-21-55-02.bag
│           └── task_pcd_test_2024-12-27-22-04-02.bag
├── msg_process.py
├── README.md
├── replay_buffer.py
├── rosbag2zarr.py
├── rosbag2zarr.sh
└── Task2-RearangeToy.json
```

### 2. run command:  
**before run command, you need modify json file `Task2-RearangeToy.json` and `config.py` according to your rosbag topic and message type.**
```bash
bash data-convert/convert_rosbag_to_zarr.sh --bag_folder_path data-example/Task2-RearangeToy/kuavo-rosbag --config Task2-RearangeToy.json
```

### 3. after convert content strucutre:
```
data-convert
├── config.py
├── data-example
│   └── Task2-RearangeToy
│       ├── kuavo-rosbag
│       ├── kuavo-zarr
│       ├── plt-check
│       ├── raw-video
│       └── sample-video
├── msg_process.py
├── README.md
├── replay_buffer.py
├── rosbag2zarr.py
├── rosbag2zarr.sh
└── Task2-RearangeToy.json
```

## train with zarr data
1. modify/add `config` in `/home/camille/IL/diffusion/diffusion_policy/config/` according to your data
    - task config: KuavoToy_task.yaml
    - training config: train_diffusion_unet_real_image_workspace_KuavoToy_task.yaml

2. modify/add `dataset` in `/home/camille/IL/diffusion/diffusion_policy/dataset/` according to your data
    - train dataset loader: /home/camille/IL/diffusion/diffusion_policy/dataset/pusht_image_dataset_KuavoToy_task.py

3. run command:
```bash
python /home/camille/IL/diffusion/train.py --config-name train_diffusion_unet_real_image_workspace_KuavoToy_task
```

## deploy(TODO)
