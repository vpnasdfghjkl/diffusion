cd /home/lab/hanxiao/dataset/kuavo/task_test
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
