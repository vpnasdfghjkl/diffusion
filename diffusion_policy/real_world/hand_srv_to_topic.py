#!/usr/bin/env python
import rospy
from dynamic_biped.msg import robot_hand_eff  # 替换为你的包名和消息文件名
import random
from kuavoSDK import kuavo
import threading
import sys

class HandSrvToTopic:
    def __init__(self):
        rospy.init_node('hand_srv_to_topic', anonymous=True)
        self.robot_instance = kuavo("3_7_kuavo")
        self.status = 0
        self.prev_status = 0
        self.input_thread = threading.Thread(target=self.read_cmd)
        self.input_thread.start()
        self.pub = rospy.Publisher('robot_hand_eff', robot_hand_eff, queue_size=10)
        self.rate = rospy.Rate(30) 
        self.publisher()

    
    def read_cmd(self):
        while (True):
            cmd = input()
            cmd = int(cmd)
            if self.prev_status != cmd:
                self.status = cmd

    def publisher(self):
        # 初始化节点
        # 创建一个publisher，发布名为'array_topic'的topic，消息类型为ArrayMessage
        # 设置发布频率

        while not rospy.is_shutdown():
            # 创建一个ArrayMessage实例
            msg = robot_hand_eff()
            msg.header.stamp = rospy.Time.now()
            
            if self.status == 0:
            # 填充数据字段，这里我们使用随机数生成一个长度为5的浮点数数组
                msg.data = [0,90,0,0,0,0,0,90,0,0,0,0]                
            elif self.status == 1:
                msg.data = [30,90,90,90,90,90,30,90,90,90,90,90]
            # 发布消息
            self.pub.publish(msg)

            if self.status != self.prev_status:
                if self.status == 0:
                    l_hand_position = [0,90,0,0,0,0]
                    r_hand_position = [0,90,0,0,0,0]
                    print("input0")
                    self.robot_instance.set_robot_end_position_control(l_hand_position, r_hand_position)
                else:

                    l_hand_position = [30,90,90,90,90,90]
                    r_hand_position = [30,90,90,90,90,90]
                    print("input1") 
                    self.robot_instance.set_robot_end_position_control(l_hand_position, r_hand_position)
                self.prev_status = self.status

            # 按照设定的频率休眠
            self.rate.sleep()

        rospy.on_shutdown(self.input_thread.join)

if __name__ == '__main__':
    nod = HandSrvToTopic()
    # HandSrvToTopic()
