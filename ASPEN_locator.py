"""
TL;DR : Python fruit locator, 3D bounder box drawer and size calculator 
Version: 0.1 - 20220523
Author: Camilo.chiang@gmail.com

    Input:
        - List of detections using a object detection algorithm (YOLOv5, Faster R-CNN)
        - Deep frame aligned to the video frame . Disparity filter is recommended
            WARNING: is recomended to use same FPS configuration for both cameras
        - Relative position and camera angle from SLAM algorithm (R3Live, FAST-LIVO)
    Output:
        - 3D bounding boxes
        - List of box coordinates
        - List of sizes

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
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray


class Detector():
    def __init__(self):
        # Subscribe
        self.sub_annotations = rospy.Subscriber("/camera/color/image_raw", Image, self.update)
        self.spin_thread = Thread(target=lambda: rospy.spin(), daemon=True)
        self.spin_thread.start()

        # And we will publishe an array
        self.pub_marker_array = MarkerArray()
    
    def update(self, msg):
        pass
    
    def __iter__(self):
        return self

    def __next__(self):
        if not self.queue_frame.empty():
            return True, self.queue_frame.get()
        else:
            return False, self.queue_stand_by_frame.get()    