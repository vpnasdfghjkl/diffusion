import rospy
import open3d as o3d
import numpy as np
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import ByteMultiArray

def compress_pointcloud_callback(msg):
    # 将 ROS 点云消息转为 Open3D 格式
    points = np.array(list(pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z", "rgb"))))
    o3d_cloud = o3d.geometry.PointCloud()
    o3d_cloud.points = o3d.utility.Vector3dVector(points[:, :3])  # xyz 坐标

    # 压缩点云：存储为 PLY 格式，并转换为二进制流
    compressed_data = o3d.io.write_point_cloud_to_buffer(o3d_cloud, format="ply")

    # 发布压缩后的点云数据
    compressed_msg = ByteMultiArray(data=compressed_data)
    compressed_pub.publish(compressed_msg)

if __name__ == "__main__":
    rospy.init_node('pointcloud_compressor')

    # 订阅原始点云数据
    rospy.Subscriber('/camera/depth/color/points', PointCloud2, compress_pointcloud_callback)

    # 发布压缩后的点云数据
    compressed_pub = rospy.Publisher('/camera/depth/color/points_compressed', ByteMultiArray, queue_size=10)

    rospy.spin()
