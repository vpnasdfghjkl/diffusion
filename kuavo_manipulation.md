![alt text](<2024-11-26 19-48-09 的屏幕截图.png>)

***roscore -> /home/lab/cam.sh -> /home/lab/hd.sh -> ssh lab@192.168.3.9 -> ssh lemon@192.168.3.27 -> /home/lab/rec.sh***

---

## **Kuavo Manipulation 操作说明**

### **1. 启动 NUC 和机器人控制系统**

#### **1.1 开机步骤**
1. **启动 NUC：** 按下急停按钮下方的 NUC 开机按钮。
2. **运行 ROS 核心节点及脚本：**
   - **PC_Terminal_1: 启动 ROS 核心**
     ```bash
     roscore
     ```
   - **PC_Terminal_2: 启动 Realsense 摄像头节点**
     
     
     ```bash
      # normal collect
      ## script launch 
         bash ~/cam.sh
      ## detail:
         roslaunch realsense2_camera rs_camera.launch align_depth:=true camera:=cam_1 serial_no:=342522073176
         roslaunch realsense2_camera rs_camera.launch align_depth:=true camera:=cam_2 serial_no:=327122077711
         rviz
         ### rviz config:File -> Recent Configs -> /home/lab/pcd.rviz
     
      # pcd collect
      roslaunch realsense2_camera rs_camera.launch color_fps:=30 color_height:=480 color_width:=640
      roslaunch realsense2_camera rs_rgbd.launch camera:=cam_2 serial_no:=342522073176
      roslaunch realsense2_camera rs_rgbd.launch camera:=cam_1 serial_no:=327122077711

      roslaunch realsense2_camera rs_camera.launch filters:=pointcloud align_depth:=true camera:=cam_1 serial_no:=342522073176
      roslaunch realsense2_camera rs_camera.launch filters:=pointcloud align_depth:=true camera:=cam_2 serial_no:=327122077711
     
     ```
   - **PC_Terminal_3: 启动 Realsense 摄像头节点**
      connect to robot:
      ssh lab@192.168.3.9  
      password:'   '(three space) 

   - **NUC_Terminal_1: 启动上肢状态控制节点**
     ```bash
     # scripts launch
     sudo su
     ./arm.sh # before launch this, fix on right initial position about shoulder and head 
         # after launch, according to the terminal output tips, press 'o','o','v' orderly

         
      ## detail
      cd /home/lab/syp/kuavo_ws/
      source devel/setup.bash
      rosrun dynamic_biped highlyDynamicRobot_node --real --cali
     ```
   - **PC_Terminal_4: 启动 Realsense 摄像头节点**
      connect to robot:
      ssh lab@192.168.3.9  
      password:'   '(three space) 
      
   - **NUC_Terminal_2: launch the script to publish eef pose with the joint 启动正解程序**

     ```bash
      # scripts launch
     sudo su
     ./fk_ik.sh # before launch this, fix on right position about shoulder and head 
         # after launch, at the forth terminal, set the head position next 
     ```

   - **NUC_Terminal_3: adjust the head position to right pos 调整头部至合适位置**
     ```bash
     cd /home/lab/syp/kuavo_ws/
     source devel/setup.bash
     conda deactivate
     rostopic pub /robot_head_motion_data dynamic_biped/robotHeadMotionData "{joint_data: [-22.0, -25.0]}"
     <!-- rostopic pub /robot_head_motion_data dynamic_biped/robotHeadMotionData "{joint_data: [-6.0, -22.0]}"  -->
     ```

     **备注:** 该命令用于 Kuavo_toy 任务的头部调整。

   **PC_Terminal_5: launch the remote control arm 连接到远程机器人并运行遥操作程序**
   ```bash
   ssh lemon@192.168.3.27
   password: softdev
   ```
   run python in the terminal 然后在远程终端运行：
   ```bash
   python /home/lemon/robot_ros_application/catkin_ws/src/DynamixelSDK/python/tests/protocol1_0/position_publish_2_for_huawei.py
   ```

   - **PC_Terminal_5: launch the hand state(open/close) topic pub node**
   ```bash
   source /home/lab/hanxiao/kuavo_ws/devel/setup.zsh
   conda activate robodiff
   python /home/lab/hanxiao/kuavo_ws/src/scripts/hand_srv_to_topic.py

   # press '1' to close, '0' for open hand 
   ```

   - **PC_Terminal_6: rosbag record 使用 `rosbag` 记录数据**














<!-- #### **1.2 启动服务和程序**
1. **PC_Terminal_4: 开启手臂服务**
   ```bash
   source /home/lab/hanxiao/kuavo_ws/devel/setup.zsh
   conda activate robodiff
   python /home/lab/hanxiao/kuavo_ws/src/scripts/hand_srv_to_topic.py
   ```
2. **PC_Terminal_3: 连接到远程机器人并运行遥操作程序**
   ```bash
   ssh lemon@192.168.3.27
   ```
   然后在远程终端运行：
   ```bash
   python /home/lemon/robot_ros_application/catkin_ws/src/DynamixelSDK/python/tests/protocol1_0/position_publish_2_for_huawei.py
   ```

#### **1.3 数据录制**
1. **PC_Terminal_5: 使用 `rosbag` 记录数据**
   ```bash
   cd /home/lab/hanxiao/dataset/kuavo
   mkdir Task2-task2name
   cd Task2-task2name
   
   rosbag record -o Kuavo_task_test \
       /camera/color/image_raw/compressed \
       /camera/depth/image_rect_raw/compressed \
       /drake_ik/cmd_arm_hand_pose \
       /drake_ik/real_arm_hand_pose \
       /kuavo_arm_traj \
       /robot_arm_q_v_tau \
       /robot_hand_eff \
       /robot_hand_position \
       --bz2 \
       --duration=20 \
       --quiet
   ```

### **2. 常用 ROS 命令**

#### **2.1 查看当前 ROS 话题**
```bash
rostopic list
```
**常见话题包括：**
- `/control_robot_hand_position` （灵巧手轨迹发布）
- `/kuavo_arm_traj` （手臂轨迹发布）
- `/robot_arm_q_v_tau` （手臂状态）
- `/robot_hand_position` （灵巧手状态）
- `/robot_head_motion_data` （头部运动数据）

#### **2.2 查看话题内容**
```bash
rostopic echo /robot_arm_q_v_tau
```
**注意:** 如果提示错误，例如：
```
ERROR: Cannot load message class for [dynamic_biped/robotArmQVVD].
```
请执行以下命令以重新加载消息：
```bash
source ~/syp/kuavo_ws/devel/setup.bash
```

### **3. 启动 Lemon 的遥操作**
1. **连接到 Lemon**
   ```bash
   ssh lemon@192.168.1.115
   ```
   **密码:** `softdev`

2. **运行遥操作程序**
   确保遥操作臂位于正确初始位置后运行：
   ```bash
   python /home/lemon/robot_ros_application/catkin_ws/src/DynamixelSDK/python/tests/protocol1_0/position_publish_2_for_huawei.py
   ```

### **4. 错误排查**

#### **常见错误提示**
1. **话题数据类型不匹配**
   ```
   [ERROR]: Client wants topic /robot_arm_q_v_tau to have datatype/md5sum [...], but our version has [...]. Dropping connection.
   ```
   **解决方案:** 检查话题的发布端与订阅端是否使用了相同的数据类型，必要时重新构建消息包。

2. **ROS 消息未加载**
   ```
   ERROR: Cannot load message class for [...]
   ```
   **解决方案:** 重新加载环境变量：
   ```bash
   source ~/syp/kuavo_ws/devel/setup.bash
   ```

#### **调试步骤**
1. 确保所有硬件（控制线和传感器）连接牢固。
2. 确认急停按钮未激活。
3. 检查运行日志并及时修复异常。

---

**备注:** 在操作过程中保持设备监控，确保硬件和软件系统的稳定性。 -->