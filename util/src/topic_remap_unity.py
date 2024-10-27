#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
from sensor_msgs.msg import NavSatFix, Imu
from std_msgs.msg import Float32
from sensor_msgs.msg import PointCloud2

from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class TopicRemap():
    def __init__(self):
        print("unity ros start")
        self.bridge = CvBridge()

        self._set_ros()
        return
    
    def _set_ros(self):
        # ROS
        rospy.init_node('topic_remap')
        self.unity_gps_sub = rospy.Subscriber('/KIMM/gps', NavSatFix, self.callback_unity_gps)
        self.unity_imu_sub = rospy.Subscriber('/KIMM/imu', Imu, self.callback_unity_imu)
        self.unity_image_sub = rospy.Subscriber('/KIMM/image/compressed', CompressedImage, self.callback_unity_image)
        self.unity_speed_sub = rospy.Subscriber('/KIMM/speedometer', Float32, self.callback_unity_speed)

        self.erp_gps_pub = rospy.Publisher('/ublox_gps/fix', NavSatFix, queue_size=10)
        self.erp_imu_pub = rospy.Publisher('/imu/data', Imu, queue_size=10)
        self.erp_image_pub = rospy.Publisher('/image_lane', Image, queue_size=10)
        self.erp_image2_pub = rospy.Publisher('/image_traffic', Image, queue_size=10)
        self.erp_speed_pub = rospy.Publisher('/cur_speed', Float32, queue_size=10)
        return
    
    def callback_unity_speed(self, speed_msg):
        new_speed = abs(speed_msg.data)
        
        self.erp_speed_pub.publish(new_speed)

    def callback_unity_gps(self, gps_msg):
        new_gps_msg = NavSatFix()
        new_gps_msg.header = gps_msg.header

        new_gps_msg.latitude = gps_msg.latitude
        new_gps_msg.longitude = gps_msg.longitude
        new_gps_msg.altitude = gps_msg.altitude
        self.erp_gps_pub.publish(new_gps_msg)

    def callback_unity_imu(self, imu_msg):
        imu_msg.header.frame_id = 'base_link'
        # imu_msg.linear_acceleration.x = 0
        # imu_msg.linear_acceleration.y = 0
        # imu_msg.linear_acceleration.z = 0
        self.erp_imu_pub.publish(imu_msg)

    def callback_unity_image(self, comp_img_msg):
        # Convert compressed image message to a numpy array
        np_arr = np.frombuffer(comp_img_msg.data, np.uint8)

        # Decode the compressed image into an OpenCV image (BGR format)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Convert OpenCV image to ROS Image message
        ros_image = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")

        # Publish the decompressed image
        self.erp_image_pub.publish(ros_image)
        self.erp_image2_pub.publish(ros_image)
    # def callback_morai_speed(self, speed_msg):
    #     self.erp_speed_pub.publish(speed_msg)

    # def callback_tl_cam(self, img_msg):
    #     self.erp_tl_cam_pub.publish(img_msg)
    
    # def callback_dl_cam(self, img_msg):
    #     self.erp_dl_cam_pub.publish(img_msg)
        
if __name__ == "__main__":
    try:
        # ROS
        tr = TopicRemap()
        rospy.spin()
        
    except rospy.ROSInterruptException:
        pass