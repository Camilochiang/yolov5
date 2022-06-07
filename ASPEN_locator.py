"""
TL;DR : Python fruit locator, 3D bounder box drawer and size calculator 
Version: 0.1 - 20220523
Author: Camilo.chiang@gmail.com

    Input:
        - Extrinsics (Rotation and translation) from depth camera to RGB camera

        - List of detections using a object detection algorithm (YOLOv5, Faster R-CNN)
        - Deep frame aligned to the video frame . Disparity filter is recommended
            WARNING: is recomended to use same FPS configuration for both cameras
        - Relative position and camera angle from SLAM algorithm (R3Live, FAST-LIVO)
    Output:
        - 3D bounding boxes
        - List of box coordinates
        - List of sizes
    
    Notes: As we use solid lidar, our SLAM algorithm are either R3Live or FAST-LIVO. In both cases they take IMU as the body frame and then treat Lidar Frame and IMU **AS the same** (https://github.com/hku-mars/r3live/issues/16#issuecomment-1004534830)

"""
# General use
import os
import numpy as np
import time
import cv2
from threading import Thread
import queue

# ROS
import rospy
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray



class Detector():
    def __init__(self):
        '''
        We will need to transform from RGB to Real word coordinates

        '''
        # Subscribe
        self.sub_annotations = rospy.Subscriber("/YOLOv5_pub", Image, self.update)
        self.camera_pos = rospy.Subscriber('/camera_odom', Odometry, self.camera_position)
        self.spin_thread = Thread(target=lambda: rospy.spin(), daemon=True)
        self.spin_thread.start()

        # And we will publishe an array
        self.pub_marker_array = rospy.Publisher('/ASPEN_3D_detections', MarkerArray)
        
        self.marker = Marker()
        self.marker.id = 0
        self.marker.header.frame_id = "/world"
        self.marker.scale.x = .1 # Default marker is 1 m
        self.marker.scale.y = .1
        self.marker.scale.z = .1
        self.marker.color.r = 1.0
        self.marker.color.g = 1.0
        self.marker.color.b = 1.0
        self.marker.color.a = 1.0
        self.marker.type = 1
        #self.marker.action = Marker.ADD
        self.marker.lifetime = rospy.Duration(0)
        self.marker.frame_locked = True   
    
    def update(self, msg):
        pass
    
    def camera_position(self, msg):
        self.marker_array = MarkerArray() # We create the marker array here to empty the it so we dont publish the same several times

        arrived_msg = msg.pose
        self.marker.id += 1
        self.marker.pose.position.x = arrived_msg.pose.position.x + 1 # Front (+)
        self.marker.pose.position.y = arrived_msg.pose.position.y  # Left (+)
        self.marker.pose.position.z = arrived_msg.pose.position.z  # Up (+)
        self.marker.pose.orientation.x = arrived_msg.pose.orientation.x + 0.0
        self.marker.pose.orientation.y = arrived_msg.pose.orientation.y + 0.0
        self.marker.pose.orientation.z = arrived_msg.pose.orientation.z + 0.0
        self.marker.pose.orientation.w = arrived_msg.pose.orientation.w + 0.0

        self.marker_array.markers.append(self.marker)

        self.pub_marker_array.publish(self.marker_array)
        #Every time that a position arrive, for each detection we will publish
        
    def __iter__(self):
        return self

    def __next__(self):
        if not self.queue_frame.empty():
            return True, self.queue_frame.get()
        else:
            return False, self.queue_stand_by_frame.get()    