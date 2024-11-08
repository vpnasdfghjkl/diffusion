rosbag record -o SongLing_task2 \
             /camera_l/color/compressed \
             /camera_f/color/compressed \
             /camera_r/color/compressed \
             /master/joint_left \
             /master/joint_right \
             /puppet/joint_left \
             /puppet/joint_right \
             --bz2 \
             --duration=20 \
             --quiet
