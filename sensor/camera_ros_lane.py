#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

def camera_example():
    bridge = CvBridge()

    # Set the desired resolution
    width, height = 640, 360

    # Set the camera's resolution
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        rospy.logerr('Failed to open video file')
        return

    rate = rospy.Rate(30)  # Publish video at 30Hz
    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if not ret:
            break

        img_msg = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        img_msg.header.stamp = rospy.Time.now()
        # img_msg = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        cam_exam.publish(img_msg)
        rate.sleep()

    cap.release()

if __name__ == '__main__':
    rospy.init_node('video_publisher', anonymous=True)
    cam_exam = rospy.Publisher('/usb_cam/image_lane', Image, queue_size=10)
    try:
        camera_example()

    except rospy.ROSInterruptException:
        pass

# cx = 405.22805
# cy = 318.98843
# fx = 789.71231
# fy = 791.33893
# distortion_model = plumb_bob
# dist = [-3.692636e-01 1.282285e-01 7.064332e-04 2.316315e-04 -1.097098e-02 0.000000e+00