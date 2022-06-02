#!/home/ubuntu/Agroscope/Python_venvironments/YOLO_v5_venv/bin/python3
from pathlib import Path
import sys
import time
import datetime
import cv2
#import math
#os.chdir('/home/ubuntu/Agroscope/ASPEN/Software/ROS/src/yolov5')

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
#ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image as Image_ROS
from visualization_msgs.msg import MarkerArray

import ASPEN_detect
import ASPEN_locator
from utils.datasets_agroscope import LoadROS

def detect_and_draw(
    weights = ROOT / 'weights/yolov5s.pt',
    conf_thres = 0.4,
    ROS_topic = "/camera/color/image_raw",
    topic_compressed = False,
    FPS = 15):
    print('\n[ASPEN   ]: Detector and polygon drawing started')

    rospy.init_node('YOLOv5', anonymous=False)
    dataset = LoadROS(sources = 'ROS',
        topic= ROS_topic,
        compressed = bool(topic_compressed),
        fps=FPS, 
        img_size=(1024,1024), 
        stride=32)

    # Now we can start the detector, who will iterate the dataset
    detector = ASPEN_detect.YOLOv5(dataset = dataset,
        weights = weights, 
        conf_thres = conf_thres)

    # And our box locator
    #locator = ASPEN_locator.Detector()
    
    pub_frame = rospy.Publisher('YOLOv5_pub', Image_ROS, queue_size=5)
    #pub_box = rospy.Publisher('ASPEN_bounding_boxes', MarkerArray, queue_size=10)
    br = CvBridge()
    
    for status, frame, count in detector:
        if not rospy.is_shutdown():
            pub_frame.publish(br.cv2_to_imgmsg(frame))
            #pub_box.publish(next(locator))
        else:
            break

    detector.close()

if __name__ == '__main__':
    try:        
        detect_and_draw(weights = rospy.get_param('YOLO_v5/weights'),
            conf_thres = float(rospy.get_param('YOLO_v5/conf_thres')),
            ROS_topic = rospy.get_param('YOLO_v5/ROS_topic'),
            topic_compressed = rospy.get_param('YOLO_v5/topic_compressed'),
            FPS = rospy.get_param('YOLO_v5/FPS'))
    except rospy.ROSInterruptException:
        pass        