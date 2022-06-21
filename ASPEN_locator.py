#!/home/ubuntu/Agroscope/Python_venvironments/YOLO_v5_venv/bin/python3
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
import time
import numpy as np
from pathlib import Path
from threading import Thread
from scipy.spatial.transform import Rotation

# ROS
import rospy
from std_msgs.msg import Float64MultiArray
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image as Image_ROS
from sensor_msgs.msg import CompressedImage
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

class Camera():
    def __init__(self, resolution, FPS, distortion, intrinsic, extrinsic_inv):
        """
            Import camera details for posterior use in python
        """
        self.resolution = resolution
        self.FPS = FPS        
        self.distortion = np.array(distortion)
        self.intrinsic = np.array(intrinsic).reshape((3,3))
        self.camera_pose =  np.identity(4) if extrinsic_inv == None else extrinsic_inv #As this will update with time and the first frame is 0,0,0, we initially pass None, but update later.
        self.deph_frame = None

        # The camera need to be transformed from cv2 coordinate system to ROS system
        #self.Cv2ROS = Rotation.from_euler('yzx',[-np.pi/2,np.pi/2,0]).as_matrix() - Is already ncluded in r3live quaternions look like

        #ROS
        self.camera_odom_sub = rospy.Subscriber('/camera_odom', Odometry, self.camera_position)
        self.dep_sub = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image_ROS, self.get_depth)
        self.spin_thread = Thread(target=lambda: rospy.spin(), daemon=True)
        self.spin_thread.start()

    def camera_position(self, msg):
        """
            Update the camera position with respect to the world
            Input:
                - Camera pose w.r.t world
                - Camera rotation w.r.t world
        """
        rot = Rotation.from_quat([msg.pose.pose.orientation.x,
                                  msg.pose.pose.orientation.y,
                                  msg.pose.pose.orientation.z,
                                  msg.pose.pose.orientation.w])
        self.camera_pose[:3,:3] = rot.as_matrix()
        self.camera_pose[:3,3] = [msg.pose.pose.position.x, 
                             msg.pose.pose.position.y, 
                             msg.pose.pose.position.z]

    def get_depth(self, msg):
        """
            Get last depth image in metters(m)
        """
        self.depth = np.frombuffer(msg.data, np.uint16).reshape( 1080, 1920)
        self.depth = self.depth*0.001

class Tracer():
    def __init__(self, Camera):
        '''
            Trace polygons for each detected object
            Input
                Camera: A camera object that include resolution, intrinsic parameters and others
                Camera_pose : Camera pose and position  w.r.t world
                Camera_point_coordinates : Image coordinates where an object was detected, in the shape of [n x 4]
        '''
        #
        self.z_values = np.array([]) # We will keep an average distance to get rid of the bad measurements due missing data
        # Camera
        self.camera = Camera

        # Subscribe
        self.sub_annotations = rospy.Subscriber("/YOLOv5/objs_xy", Float64MultiArray, self.trace)
        self.spin_thread = Thread(target=lambda: rospy.spin(), daemon=True)
        self.spin_thread.start()

        # And we will publishe an array
        self.pub_marker_array = rospy.Publisher('/ASPEN/3D_detections', MarkerArray, queue_size = 10)
        # We will have up to 4 categories
        self.markers_colors = [[0.180, 0.349, 0.141],[0.749, 0.745, 0.192],[0.949, 0.498, 0.070],[0.752, 0.137, 0.066]]
        
        self.marker = Marker()
        self.marker.id = 0
        self.marker.header.frame_id = "/world"
        self.marker.scale.x = .08 # Default marker is 1 m
        self.marker.scale.y = .08
        self.marker.scale.z = .08
        self.marker.color.r = self.markers_colors[0][0]
        self.marker.color.g = self.markers_colors[0][1]
        self.marker.color.b = self.markers_colors[0][2]
        self.marker.color.a = 0.75
        self.marker.type = 1
        self.marker.pose.orientation.x = 0.0
        self.marker.pose.orientation.y = 0.0
        self.marker.pose.orientation.z = 0.0
        self.marker.pose.orientation.w = 1.0
        #self.marker.action = Marker.ADD
        self.marker.lifetime = rospy.Duration(0)
        self.marker.frame_locked = True   

        # Where will be save the output?
        self.directory = Path(os.getcwd()).parent
        self.file = str(self.directory) + '/' + time.strftime("%Y%m%d_%H%M%S") + '_results.txt'
        self.file = open(self.file, 'w')
        self.file.write('x1\ty1\tx2\ty2\tPosition\tCategory\tCoordinates\n')

    def trace(self, msg):
        """
            Publish an array of markers for each detected object
        """
        self.marker_array = MarkerArray()
        new_data = msg.data
        new_data = np.array(new_data).reshape(int(len(new_data)/7),7)
        for detection in new_data:
            # Detection is a list of x1,y1,x2,y2,track_id,last seen probability, category
            # We have to calculate:
            #   Pixel to point-wrt-camera-sys-cv2
            #   point-sys-cv2 to point-sys-ROS
            #   point-sys-ROS to point-wrt-
            
            # Each detection bring 2 coorners and the other two will be calculataed
            # 0. We need to get z before any calculations
            z_value = self.camera.depth[int(detection[1]):int(detection[3]),int(detection[0]):int(detection[2])] # Numpy array so is y1:yn,x1;xn order
            mask = np.logical_and(z_value > 0 , z_value < 10) # We will limit my data to 10 as I should not be further away than that ever
            if len(z_value[mask]) > 0:
                z_value = np.min(z_value[mask])

                # We will plot only if the distance is not 50% higher than the average
                if self.marker.id < 3 and z_value:
                    self.z_values = np.append(self.z_values, z_value)
                    Trace = True
                else:
                    if abs(np.mean(self.z_values) - z_value) < 1.5*np.mean(self.z_values):
                        self.z_values = np.append(self.z_values, z_value)
                        Trace = True
                    else:
                        Trace = False

                if Trace:
                    self.marker.color.r = self.markers_colors[int(detection[6])][0] 
                    self.marker.color.g = self.markers_colors[int(detection[6])][1] 
                    self.marker.color.b = self.markers_colors[int(detection[6])][2] 

                    # 1. We pass from pixel to camera distances using the intrinsic and deep, then we pass to ROS using the predefined transformation
                    xy_left_top = np.array([detection[0], detection[1], 1])
                    xy_left_top = (np.linalg.inv(self.camera.intrinsic) @ xy_left_top * z_value) #@ self.camera.Cv2ROS
                    xy_rightbottom = np.array([detection[2], detection[3], 1])
                    xy_rightbottom = (np.linalg.inv(self.camera.intrinsic) @ xy_rightbottom * z_value)# @ self.camera.Cv2ROS

                    xy_left_top = (self.camera.camera_pose @ np.append(xy_left_top, 1))[:3]

                    # The pivot point of a cube is at the center of it.
                    self.marker.pose.position.x =  xy_left_top[0]
                    self.marker.pose.position.y =  xy_left_top[1]
                    self.marker.pose.position.z =  xy_left_top[2]
                    self.marker_array.markers.append(self.marker) 

                    # Write to file. So basically if there is no depth we will no have it later in the file
                    for item in detection:
                        self.file.write(str(item) + '\t')
                    self.file.write('\n')
                    self.marker.id += 1        
            
        self.pub_marker_array.publish(self.marker_array)


    def __iter__(self):
        return self

    def __next__(self):
        if not self.queue_frame.empty():
            return True, self.queue_frame.get()
        else:
            return False, self.queue_stand_by_frame.get()    