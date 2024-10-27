#!/usr/bin/env python3

import numpy as np
import time

from sensor_fusion_handler import *

# ROS
import rospy
from cv_bridge import CvBridge
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose, PoseArray
from sensor_msgs.msg import Image
from std_msgs.msg import String

import message_filters

# 올해 파라미터
# self.intrinsic = np.array([[381.18310547,   0.,         319.02992935, 0.],
#                             [0. ,        448.0055542, 207.31755552, 0.],
#                             [0.000000, 0.000000, 1.000000, 0.]])

# self.extrinsic = self.rtlc(alpha = np.radians(38.0),
#                                 beta = np.radians(31.3),
#                                 gamma = np.radians(4.1), 
#                                 tx = 0.965, ty = -0.23, tz = -0.95)

# self.intrinsic_right = np.array([[378.68261719,   0.,         328.19930137, 0.],
#                     [0. ,        443.68624878, 153.57524293, 0.],
#                     [0.000000, 0.000000, 1.000000, 0.]])
# self.extrinsic_right = self.rtlc(alpha = np.radians(30.8),
#                                     beta = np.radians(-33.2),
#                                     gamma = np.radians(2.3), 
#                                     tx = 0.965, ty = 0.218, tz = -0.95)
        
class SensorFusion():
    def __init__(self):
        self.bridge = CvBridge()
        # self.intrinsic = np.array([[381.18310547,   0.,         319.02992935, 0.],
        #                     [0. ,        448.0055542, 207.31755552, 0.],
        #                     [0.000000, 0.000000, 1.000000, 0.]])
        
        # 트랙주행 왼쪽 카메라
        # self.intrinsic = np.array([[381.18310547,   0.,         319.02992935, 0.],
        #                     [0. ,        448.0055542, 207.31755552, 0.],
        #                     [0.000000, 0.000000, 1.000000, 0.]])

        # 본선 중앙 카메라
        self.intrinsic = np.array([[470.25, 0.0, 323.75, 0.0],
                                    [0.0, 325.17, 241.65, 0.0],
                                    [0.0, 0.0, 1.0, 0.0]])
        
        self.extrinsic = self.rtlc(alpha = np.radians(6.3),
                                        beta = np.radians(0.4),
                                        gamma = np.radians(0.0), 
                                        tx = 0.642, ty = 0.0, tz = -0.402)
        
        self.clusters_2d = None
        self.bboxes = None

        # ROS
        rospy.init_node('sensor_fusion_person', anonymous=True)

        self.image_sub = rospy.Subscriber("/yolo/person", Image, self.callback_img)

        self.cluster_sub = message_filters.Subscriber('/adaptive_clustering/markers', MarkerArray)
        self.bbox_sub = message_filters.Subscriber("/bounding_boxes/person", PoseArray)
        
        self.sync = message_filters.ApproximateTimeSynchronizer([self.cluster_sub, self.bbox_sub], queue_size=10, slop=0.5, allow_headerless=True)
        self.sync.registerCallback(self.callback_fusion)
        
        self.result_img_pub = rospy.Publisher("/sensor_fusion/result_img",Image,queue_size=10)
        self.person_pub = rospy.Publisher('/sensor_fusion/person',PoseArray,queue_size=10)

        print("\033[1;33m Starting camera and 3D LiDAR sensor fusion. \033[0m")
        return
    
    def callback_img(self, img_msg):
        bridge = CvBridge()
        img = bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        if self.clusters_2d is not None and self.bboxes is not None:
            visualize_cluster_2d(self.clusters_2d, img)
            visualize_bbox(self.bboxes, img)

        result_img = bridge.cv2_to_imgmsg(img, encoding="bgr8") 
        result_img.header.stamp = rospy.Time.now()
        self.result_img_pub.publish(result_img)

    def callback_fusion(self, cluster_msg, bbox_msg):
        first_time = time.perf_counter()

        # Clustering points to np array
        clusters = cluster_for_fusion(cluster_msg) # 클러스터링 중점을 계산 (3D)
        
        # 2D bounding boxes
        bboxes, bboxes_label = bounding_boxes(bbox_msg)
        self.bboxes = bboxes

        # 3D BBOX to Pixel Frame
        clusters_2d, valid_indicies = projection_3d_to_2d(clusters, self.intrinsic, self.extrinsic)
        self.clusters_2d = clusters_2d

        # Sensor Fusion (Hungarian Algorithm)
        matched = hungarian_match(clusters_2d, bboxes, bboxes_label, distance_threshold=50)

        labels = get_label(matched, valid_indicies)
        print("라벨 개수: ", len(labels), ", ", len(clusters.T[:,:3]))
        
        target_clusters = []
        person_pose_array = PoseArray()
        person_pose_array.header.frame_id = 'velodyne'
        person_pose_array.header.stamp = rospy.Time.now()
        for idx, id in enumerate(labels):
            if id == 0: # person label
                target_clusters.append(clusters.T[:,:3][idx])
        
        print(target_clusters)
        if target_clusters:
            # 각 클러스터 좌표에 대해 Pose를 생성하여 PoseArray에 추가
            for cluster in target_clusters:
                pose = Pose()
                pose.position.x = cluster[0]  # x 좌표
                pose.position.y = cluster[1]  # y 좌표
                pose.position.z = cluster[2]  # z 좌표

                # PoseArray에 Pose 추가
                person_pose_array.poses.append(pose)
        
            self.person_pub.publish(person_pose_array)

        print("소요 시간: {:.5f}".format(time.perf_counter() - first_time))
        print("")
        return
    
    def rtlc(self, alpha, beta, gamma, tx, ty, tz):              
        Rxa = np.array([[1, 0, 0,0],
                        [0, np.cos(alpha), -np.sin(alpha),0],
                        [0, np.sin(alpha), np.cos(alpha),0],
                        [0,0,0,1]])

        Ryb = np.array([[np.cos(beta), 0, np.sin(beta),0],
                        [0, 1, 0,0],
                        [-np.sin(beta), 0, np.cos(beta),0],
                        [0,0,0,1]])

        Rzg = np.array([[np.cos(gamma), -np.sin(gamma), 0, 0],
                [np.sin(gamma), np.cos(gamma), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]])
        
        Ry90 = np.array([[np.cos(np.deg2rad(-90)), 0, np.sin(np.deg2rad(-90)),0],
                         [0, 1, 0,0],
                         [-np.sin(np.deg2rad(-90)), 0, np.cos(np.deg2rad(-90)),0],
                         [0,0,0,1]])
                 
        Rx90= np.array([[1, 0, 0,0],
                        [0, np.cos(np.deg2rad(90)), -np.sin(np.deg2rad(90)),0],
                        [0, np.sin(np.deg2rad(90)), np.cos(np.deg2rad(90)),0],
                        [0,0,0,1]])
        
        T = np.array([[1, 0, 0, tx],      
                      [0, 1, 0, ty],                  
                      [0, 0, 1, tz],                       
                      [0, 0, 0, 1]]) 
        
        rtlc = Rzg@Rxa@Ryb@Ry90@Rx90@T
        return rtlc
    
    def make_marker(self, color):
        marker = Marker()
        marker.action = marker.ADD
        marker.type = marker.POINTS
        marker.header.frame_id = "velodyne"
        marker.header.stamp = rospy.Time.now()
        marker.lifetime = rospy.Duration(0.1)
        marker.id = int((color[0] + color[1] + color[2]) * 10000)
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.a = 1.0
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.pose.orientation.w = 1.0
        return marker
    
    @staticmethod
    def make_pose_array(points):
        pose_array = PoseArray()
        pose_array.header.stamp = rospy.Time.now()
        pose_array.header.frame_id = 'yolo'
        for x, y in points:
            pose = Pose()
            pose.orientation.x = x
            pose.orientation.y = y
            
            pose_array.poses.append(pose)
        return pose_array
    
if __name__ == '__main__':
    sensor_fusion = SensorFusion()
    rospy.spin()