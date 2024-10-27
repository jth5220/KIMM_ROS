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

WEIGHTS = 'weights/delivery_t.pt'
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
        rospy.init_node('delivery_sign_detection', anonymous=True)

        self.detected_pub = rospy.Publisher("/yolo/delivery", Image, queue_size=10)
        self.delivery_pub = rospy.Publisher("/delivery_sign", Int32MultiArray, queue_size=10)
        self.boungingboxes_pub= rospy.Publisher('/bounding_boxes/deliver', PoseArray, queue_size=10)
        self.target_sign_pub = rospy.Publisher('/target_sign',Int32, queue_size=3)
        
        self.mode_sub = rospy.Subscriber("/driving_mode", String, self.mode_cb)
        self.image_sub = rospy.Subscriber("/image_delivery", Image, self.image_cb)
        
        self.img_width = IMG_SIZE
        self.mode = None
        self.sign = 0
        self.target_sign = None
        self.stop_mode = False  # Added to control stop mode

        self.B1 = []
        self.B2 = []
        self.B3 = []

        self.queue_list = [[-1 for i in range(QUEUE_SIZE)] for j in range(len(CLASS_MAP))]
        self.id_to_queue_list = [self.queue_list[i] for i in range(len(CLASS_MAP)) for _ in range(len(CLASS_MAP[i]))]

        rospy.Timer(rospy.Duration(0.1), self.yolo_pub)

    def mode_cb(self, msg):
        self.mode = msg.data

    def image_cb(self, img, event=None):
        if self.stop_mode:  # Stop processing if in stop mode
            return

        self.img_width = img.width
        # check_requirements(exclude=('pycocotools', 'thop'))
        with torch.no_grad():
            bridge = CvBridge()
            cap = bridge.imgmsg_to_cv2(img, desired_encoding="bgr8")

            poses = PoseArray()
            poses.header.stamp = rospy.Time.now()
            poses.header.frame_id = 'yolo'
            if self.mode == 'delivery_start' or self.mode == 'delivery_finish':
                result, coords = self.detect(cap)
                image_message = bridge.cv2_to_imgmsg(result, encoding="bgr8")
                
                for coord in coords:
                    pose = Pose()
                    # Set the position of the pose
                    pose.position.x = coord[0] # 0 blue or 1 yellow
                    # pose.position.y = 0. # confidence
                    # pose.position.z = 0.
                        
                    pose.orientation.x = coord[1] #xmin
                    pose.orientation.y = coord[2] #ymin
                    pose.orientation.z = coord[3] #xmax
                    pose.orientation.w = coord[4] #ymax

                    poses.poses.append(pose)
                self.boungingboxes_pub.publish(poses)
                
            else:
                image_message = bridge.cv2_to_imgmsg(cap, encoding="bgr8")

            image_message.header.stamp = rospy.Time.now()
            self.detected_pub.publish(image_message)

    def detect(self, frame):
        img0 = frame
        img = letterbox(img0, imgsz, stride=stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        t0 = time_synchronized()
        pred = model(img, augment=AUGMENT)[0]

        pred = non_max_suppression(pred, CONF_THRES, IOU_THRES, classes=CLASSES, agnostic=AGNOSTIC_NMS)

        det = pred[0]
        numClasses = len(det)

        s = ''
        s += '%gx%g ' % img.shape[2:]

        coords = []
        if numClasses:
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            for *xyxy, conf, cls in reversed(det):
                id = int(cls)
                label = f'{names[id]} {conf:.2f}'
                plot_one_box(xyxy, img0, label=label, color=colors[id], line_thickness=3)
                xmin, ymin, xmax, ymax = [int(tensor.item()) for tensor in xyxy]

                xmean = (xmin + xmax) / 2
                ymean = (ymin + ymax) / 2

                coords.append([id, xmin, ymin, xmax, ymax])

                if xmean > 50 and xmean < self.img_width - 50:
                    if id in (0, 1, 2):
                        self.id_to_queue_list[id + 3].append(int(xmean))
                        self.id_to_queue_list[0].append(id)

                        if self.target_sign is None:
                            if id == 0:
                                self.target_sign = 3
                            elif id == 1:
                                self.target_sign = 4
                            elif id == 2:
                                self.target_sign = 5
                            
                        # Set the mode to 'delivery start' when detecting A1, A2, A3
                        print(f"Delivery start mode activated. Detected sign: {names[id]}")
                        print(f"Delivery start mode activated. Target sign: {names[self.target_sign]}")

                        if xmean >= 550:
                            print(f'Start sign A{self.sign} detected Stop.')
                            self.start_detected = True
                            # self.stop_mode = True
                            
                    self.target_sign_pub.publish(self.target_sign)
                    
                    if id == self.target_sign:
                        print(f"Target sign B{self.sign} detected.")
                        self.target_detected = True

                        if xmean >= 550:
                            print(f'Target sign B{self.sign} detected Stop.')
                            # self.stop_mode = True

                else:
                    for queue in self.queue_list:
                        if len(queue) == QUEUE_SIZE:
                            queue.append(-1)

            if len(self.B1):
                bx = min(self.B1)
                self.id_to_queue_list[6].append(bx)
            if len(self.B2):
                bx = min(self.B2)
                self.id_to_queue_list[7].append(bx)
            if len(self.B3):
                bx = min(self.B3)
                self.id_to_queue_list[8].append(bx)

            self.B1 = []
            self.B2 = []
            self.B3 = []

        else:
            for queue in self.queue_list:
                if len(queue) == QUEUE_SIZE:
                    queue.append(-1)
                    
        return img0, coords

    def delivery_vote(self, queue):
        if queue.count(-1) > int(QUEUE_SIZE / 2):
            return 0
        else:
            for element in queue:
                if element != -1:
                    val = element
            return val

    def hard_vote(self, queue):
        return statistics.mode(queue)

    def yolo_pub(self, event):
        final_check = Int32MultiArray()

        for queue in self.queue_list:
            while len(queue) != QUEUE_SIZE:
                del queue[0]

        queue_list = self.queue_list

        for idx in range(len(queue_list)):
            if idx == 0:
                final_check.data.append(self.hard_vote(queue_list[idx]))
            else:
                final_check.data.append(self.delivery_vote(queue_list[idx]))

        if final_check.data[0] != -1:
            self.sign = final_check.data[0] + 1

        if self.sign:
            pass
        else:
            print('Not Detected yet')

        self.delivery_pub.publish(final_check)

    # def process_frame(self, frame):
    #     # 프레임 크기 구하기
    #     height, width, _ = frame.shape
        
    #     # 정수로 변환하여 슬라이스 인덱스 문제 해결
    #     height = int(height)
    #     width = int(width)
        
    #     # 화면을 4등분
    #     half_height = height // 2
    #     half_width = width // 2
        
    #     bottom_left = frame[half_height:height, 0:half_width]  # 왼쪽 아래

    #     # 그레이스케일 변환
    #     gray = cv2.cvtColor(bottom_left, cv2.COLOR_BGR2GRAY)
        
    #     # 색 범위 설정 (흰색의 범위 확장)
    #     lower_white = np.array([120, 120, 150], dtype=np.uint8)
    #     upper_white = np.array([255, 255, 255], dtype=np.uint8)
    #     mask = cv2.inRange(bottom_left, lower_white, upper_white)

    #     # Contour (윤곽선) 검출
    #     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
    #     for contour in contours:
    #         area = cv2.contourArea(contour)
    #         if 6000 > area > 1500:  # 면적 임계값 설정 (예: 1000 이상의 면적)
    #             # 윤곽선을 둘러싸는 최소 면적 회전 사각형 구하기
    #             rect = cv2.minAreaRect(contour)
    #             box = cv2.boxPoints(rect)
    #             box = np.int0(box)
                
    #             # 사각형의 너비와 높이 구하기
    #             rect_width = min(rect[1])
    #             rect_height = max(rect[1])
                
    #             # 너비와 높이의 비율로 직사각형 두께 확인
    #             if 3 <= rect_height/rect_width <= 10:
    #                 # 주축의 기울기 계산 (회전각)
    #                 angle = rect[2]
                    
    #                 # 기울기가 10도 이상일 때만 영역 표시
    #                 if -50 < angle < -10:
    #                     cv2.drawContours(bottom_left, [box], -1, (0, 255, 0), 2)
    #                     print(f"넓이: {area}, 각도: {angle:.2f}, 비율: {rect_height/rect_width:.2f}")
        
    #     # 원본 프레임에 왼쪽 아래 영역을 다시 대입
    #     frame[half_height:height, 0:half_width] = bottom_left

    #     # 마스킹된 화면을 별도의 창에 표시
    #     cv2.imshow('Masked View', mask)

    #     return frame
    
def main(args=None):
    yolo = YOLOv7()
    rospy.spin()

if __name__ == "__main__":
    main()