#!/home/ubuntu/Agroscope/Python_venvironments/YOLO_v5_venv/bin/python3
from pathlib import Path
import sys
import signal
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image as Image_ROS
from std_msgs.msg import Float64MultiArray


import ASPEN_detect
import ASPEN_locator
from utils.datasets_agroscope import LoadROS
from utils.general import colorstr

def signal_handler(sig, frame):
    print(colorstr('[ASPEN - NODE   ]: Detector and polygon drawing closed'))
    global detector
    detector.close()
    sys.exit()

def detect_and_draw(
    weights = ROOT / 'weights/yolov5s.pt',
    conf_thres = 0.4,
    ROS_topic = "/camera/color/image_raw",
    topic_compressed = False,
    camera_width = 1920,
    camera_height = 1080,
    camera_FPS = 15,
    camera_distortion = [0,0,0,0,0],
    camera_intrinsics = [1370.163330078125, 0.0, 944.8233032226562, 0.0,  1368.414794921875, 552.1774291992188, 0.0, 0.0, 1.0 ],
    camera_ext_R = None, # Not to be confuse with the Camera-Lidar ext matrix.
    camera_ext_t = None,  # Not to be confuse with the Camera-lidar ext matrix
    tracker_sense = True # Default direction is moving from west to east. That means that detections move from the left side to the right side of an image
):
    print(colorstr('\n[ASPEN - NODE    ]: Detector and polygon drawing started'))

    # As we are using ROS launch, we need a good way to stop python
    signal.signal(signal.SIGINT, signal_handler)

    rospy.init_node('YOLOv5', anonymous=False)
    # Our Camera without ext as this will depend of SLAM algorithm
    ASPEN_camera = ASPEN_locator.Camera(resolution = [camera_width,camera_height],
                                        FPS = camera_FPS,
                                        distortion = camera_distortion,
                                        intrinsic = camera_intrinsics,
                                        extrinsic_inv = None)

    dataset = LoadROS(sources = 'ROS',
        topic= ROS_topic,
        compressed = bool(topic_compressed),
        fps=camera_FPS, 
        img_size=(1024,1024), 
        stride=32)

    # Now we can start the detector, who will iterate the dataset
    global detector
    detector = ASPEN_detect.YOLOv5(dataset = dataset,
        weights = weights, 
        conf_thres = conf_thres,
        tracker_sense = tracker_sense)

    # And our box tracer
    Tracer = ASPEN_locator.Tracer(Camera = ASPEN_camera)
    
    pub_frame = rospy.Publisher('/YOLOv5/out_image', Image_ROS, queue_size=5)
    pub_positions = rospy.Publisher('/YOLOv5/objs_xy', Float64MultiArray, queue_size=5)
    br = CvBridge()
    
    for status, frame, online_targets, count in detector:
        if not rospy.is_shutdown():
            #Publish objects detection
            pub_frame.publish(br.cv2_to_imgmsg(frame))
            # The global position and localizator will then update in locator
            if online_targets:
                data_to_send = Float64MultiArray()
                data_to_send.data = [x for xs in online_targets for x in xs] # A flat list that will be recover its dimention later in locator (n x 7)
                pub_positions.publish(data_to_send)
        else:
            break

if __name__ == '__main__':
    try:        
        detect_and_draw(weights = rospy.get_param('YOLO_v5/weights'),
            conf_thres = float(rospy.get_param('YOLO_v5/conf_thres')),
            ROS_topic = rospy.get_param('YOLO_v5/ROS_topic'),
            topic_compressed = rospy.get_param('YOLO_v5/topic_compressed'),
            camera_width =  rospy.get_param('r3live_vio/image_width'),
            camera_height =  rospy.get_param('r3live_vio/image_height'),
            camera_FPS = rospy.get_param('YOLO_v5/FPS'),
            camera_distortion =  rospy.get_param('r3live_vio/camera_dist_coeffs'),
            camera_intrinsics =  rospy.get_param('r3live_vio/camera_intrinsic'),
            camera_ext_R = None,
            camera_ext_t = None,
            tracker_sense = rospy.get_param('YOLO_v5/tracker_right2left'))
    except rospy.ROSInterruptException:
        pass