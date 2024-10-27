#!/usr/bin/env python3

import time
import torch
import statistics
import numpy as np

from numpy import random
from cv_bridge import CvBridge
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, check_requirements, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray, String, Int32
from geometry_msgs.msg import Pose, PoseArray

WEIGHTS = 'weights/yolov7_training.pt'
IMG_SIZE = 640
DEVICE = ''
AUGMENT = False
CONF_THRES = 0.85
IOU_THRES = 0.45
CLASSES = None
AGNOSTIC_NMS = False

QUEUE_SIZE = 13
CLASS_MAP = (
    ("id_0", "id_1", "id_2"),
    ("A1",),
    ("A2",),
    ("A3",),
    ("B1",),
    ("B2",),
    ("B3",)
)

device = select_device(DEVICE)
half = device.type != 'cpu'
print('device:', device)

model = attempt_load(WEIGHTS, map_location=device)
stride = int(model.stride.max())
imgsz = check_img_size(IMG_SIZE, s=stride)
if half:
    model.half()

names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

if device.type != 'cpu':
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

class YOLOv7:
    def __init__(self):
        rospy.init_node('person_detection', anonymous=True)

        self.detected_pub = rospy.Publisher("/yolo/person", Image, queue_size=10)
        self.person_pub= rospy.Publisher('/bounding_boxes/person', PoseArray, queue_size=10)
        
        self.image_sub = rospy.Subscriber("/image_lane", Image, self.image_cb)
        self.driving_mode_sub = rospy.Subscriber("/driving_mode", String, self.callback_driving_mode)

        self.img_width = IMG_SIZE

        self.detect_mode = False

    def callback_driving_mode(self, mode_msg):
        if mode_msg.data == 'tunnel':
            self.detect_mode = True
        else:
            self.detect_mode = False
        return
    
    def image_cb(self, img, event=None):
        self.img_width = img.width

        if self.detect_mode:
            # check_requirements(exclude=('pycocotools', 'thop'))
            with torch.no_grad():
                bridge = CvBridge()
                cap = bridge.imgmsg_to_cv2(img, desired_encoding="bgr8")

                result_img, persons = self.detect(cap)

                if persons is not None:
                    # ROS publish
                    poses = PoseArray()
                    poses.header.stamp = rospy.Time.now()
                    poses.header.frame_id = 'yolo'
                    
                    for sublist in persons:
                        pose = Pose()
                        # Set the position of the pose
                        pose.position.x = sublist[0] # class
                        pose.position.y = sublist[5] # confidence
                            
                        pose.orientation.x = sublist[1] #xmin
                        pose.orientation.y = sublist[2] #ymin
                        pose.orientation.z = sublist[3] #xmax
                        pose.orientation.w = sublist[4] #ymax

                        poses.poses.append(pose)
                        
                    self.person_pub.publish(poses)

                result_img_msg = bridge.cv2_to_imgmsg(result_img, encoding="bgr8")
                result_img_msg.header.stamp = rospy.Time.now()
                self.detected_pub.publish(result_img_msg)
        else:
            self.detected_pub.publish(img)

    def detect(self, img0):
        # # Load image
        # img0 = frame

        # Padded resize
        img = letterbox(img0, imgsz, stride=stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        # t0 = time_synchronized()
        pred = model(img, augment=AUGMENT)[0]

        # Apply NMS
        pred = non_max_suppression(pred, CONF_THRES, IOU_THRES, classes=CLASSES, agnostic=AGNOSTIC_NMS)

        # Process detections
        det = pred[0]
        numClasses = len(det)

        if numClasses:
            # Rescale boxes from img_size to img0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            persons =[]
            
            # Write results
            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=1)
                xmin, ymin, xmax, ymax = [int(tensor.item()) for tensor in xyxy]
                
                width = xmax - xmin
                height = ymax - ymin

                print('person detected. / width:', width)
                if (names[int(cls)] == 'person') and (width > 25):
                    person =[int(cls), xmin, ymin, xmax, ymax, conf]
                    persons.append(person)

            return img0, persons

        return img0, None
    
def main(args=None):
    yolo = YOLOv7()
    rospy.spin()

if __name__ == "__main__":
    main()