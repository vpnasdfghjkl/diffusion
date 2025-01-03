#!/bin/bash

# 打印调试信息
echo "Running rosbag2zarr.py with the following parameters:"

# 获取当前脚本所在的路径
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# 默认值
BAG_FOLDER_PATH="$SCRIPT_DIR/data-example/Task2-RearangeToy/kuavo-rosbag"
CONFIG_FILE="$SCRIPT_DIR/Task2-RearangeToy.json"

# 解析命令行参数
while getopts "b:c:" opt; do
  case $opt in
    b) BAG_FOLDER_PATH="$OPTARG" ;;  # 设置 bag 文件夹路径
    c) CONFIG_FILE="$OPTARG" ;;  # 设置配置文件路径
    \?) echo "Usage: $0 [-b bag-folder-path] [-c config-file]"
        exit 1 ;;
  esac
done

# 打印最终的参数信息
echo "--------------------------------------------"
echo -e "\033[1;34mRunning rosbag2zarr.py with the following parameters:\033[0m"
echo -e "\033[1;32m  bag-folder-path: $BAG_FOLDER_PATH\033[0m"
echo -e "\033[1;32m  config-file: $CONFIG_FILE\033[0m"
echo "--------------------------------------------"

# 确保 bag 文件夹路径和配置文件路径是有效的
if [[ -z "$BAG_FOLDER_PATH" || -z "$CONFIG_FILE" ]]; then
  echo "Error: Missing required parameters."
  echo "Usage: $0 [-b bag-folder-path] [-c config-file]"
  exit 1
fi

# 运行 rosbag2zarr.py 脚本
python "$SCRIPT_DIR/rosbag2zarr.py" --bag_folder_path "$BAG_FOLDER_PATH" --config "$CONFIG_FILE"
