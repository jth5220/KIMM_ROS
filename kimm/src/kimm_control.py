#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist
from ackermann_msgs.msg import AckermannDrive

from pid import PID
import numpy as np

MAX_STEER = 70
MIN_STEER = -70

class KIMMControl():
    def __init__(self):
        self.__set_attribute__()
        self.__set_ros__()
        return
    
    def __set_ros__(self):
        # ROS
        rospy.init_node('kimm_control')

        # Subscriber
        self.cmd_sub = rospy.Subscriber('/erp_command', AckermannDrive, self.callback_cmd)
        self.speeed_sub = rospy.Subscriber('/cur_speed', Float32, self.callback_speed)

        # Publisher
        self.cmd_pub = rospy.Publisher('/KIMM/cmd', Twist, queue_size=10)

        # Timer
        self.timer_driving = rospy.Timer(rospy.Duration(0.1), self.callback_driving)

    def __set_attribute__(self):
        self.car_cmd = {'throttle':0.0,
                        'steering_angle':0.0,
                        'brake':1.0,
                        'gear':0,
                        'target_speed':0.0}
        
        self.car_status = {'speed':0.0}

        self.pid = PID(kp = 1.0, ki = 0.0, kd = 0.0)
        return
    
    def callback_speed(self, speed_msg):
        self.car_status['speed'] = abs(speed_msg.data)
        return
    
    def callback_cmd(self, cmd_msg):
        self.car_cmd['target_speed'] = cmd_msg.speed
        self.car_cmd['steering_angle'] = max(min(np.radians(cmd_msg.steering_angle) * 5.0, np.radians(MAX_STEER)), -np.radians(MAX_STEER))
        self.car_cmd['brake'] = cmd_msg.jerk
        if self.car_cmd['brake'] <= 1.0:
            self.car_cmd['brake'] = 0
        return
    
    def callback_driving(self, event):
        target_speed = 0.0
        if self.car_cmd['target_speed'] > 0:
            target_speed = self.car_cmd['target_speed']
            self.car_cmd['gear'] = 0

            self.car_cmd['throttle'], pid = self.pid.update(target_speed, self.car_status['speed'])
            print("p term:",pid[0])

            if self.car_cmd['throttle'] < 0:
                error = abs(self.car_cmd['throttle'])
                self.car_cmd['throttle'] = 0.0

                if error < 0.1:
                    self.car_cmd['brake'] = 0.3
                elif error < 0.5:
                    self.car_cmd['brake'] = 0.5
                else:
                    self.car_cmd['brake'] = 1.0

            self.car_cmd['throttle'] = min(max(self.car_cmd['throttle'], 0), 0.5)
            # self.car_cmd['throttle'] *= 0.2

        elif self.car_cmd['target_speed'] < 0:
            target_speed = -self.car_cmd['target_speed']
            self.car_cmd['gear'] = 1
            
            self.car_cmd['throttle'], _ = self.pid.update(target_speed, self.car_status['speed'])

            if self.car_cmd['throttle'] < 0:
                error = abs(self.car_cmd['throttle'])
                self.car_cmd['throttle'] = 0.0
                if error < 0.1:
                    self.car_cmd['brake'] = 0.3
                elif error < 0.5:
                    self.car_cmd['brake'] = 0.5
                else:
                    self.car_cmd['brake'] = 1.0

            self.car_cmd['throttle'] = min(max(self.car_cmd['throttle'], 0), 0.5)
            # self.car_cmd['throttle'] *= 0.2

        else:
            target_speed = 0
            self.car_cmd['brake'] = 1
            self.car_cmd['throttle'] = 0

        print('throttle:', self.car_cmd['throttle'])
        print('steer:', self.car_cmd['steering_angle'])
        print('brake:', self.car_cmd['brake'])
        print('gear:', self.car_cmd['gear'])
        print("======================")
        # Cmd Publish
        cmd_msg = Twist()
        cmd_msg.linear.x = self.car_cmd['throttle']
        cmd_msg.linear.y = self.car_cmd['steering_angle']
        cmd_msg.linear.z = self.car_cmd['brake']
        cmd_msg.angular.x = self.car_cmd['gear']
        self.cmd_pub.publish(cmd_msg)
        
if __name__ == "__main__":

    try:
        # ROS
        kimm_control = KIMMControl()
        rospy.spin()

    except rospy.ROSInterruptException:
        pass