# relative note

## image_transport
```sh
mkdir -p ~/hx/image_transport_ws/src
cd ~/hx/image_transport_ws/src
catkin_create_pkg img_transport_pkg sensor_msgs cv_bridge image_transport # or catkin_create_pkg img_transport_pkg
mkdir -p img_transport_pkg/launch
echo '<launch>
  <node name="camera_f_compress" pkg="image_transport" type="republish" output="screen" args="raw in:=/camera_f/color/image_raw compressed out:=/camera_f/color" />
  <node name="camera_l_compress" pkg="image_transport" type="republish" output="screen" args="raw in:=/camera_l/color/image_raw compressed out:=/camera_l/color" />
  <node name="camera_r_compress" pkg="image_transport" type="republish" output="screen" args="raw in:=/camera_r/color/image_raw compressed out:=/camera_r/color" />
</launch>' > img_transport_pkg/compress_images.launch
cd ~/hx/image_transport_ws
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
source devel/setup.zsh  # or source devel/setup.zsh
roslaunch img_transport_pkg compress_images.launch
```