# YOLOv5 🚀 by Ultralytics, GPL-3.0 license


from concurrent.futures import thread
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from threading import Thread
from queue import Queue

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
print(f'Adding: {ROOT} to the path')
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend

from utils.datasets_agroscope import IMG_FORMATS, VID_FORMATS, LoadROSBAG, LoadROS
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from sort.sort import *
import rospy

@torch.no_grad()

class YOLOv5():
    def __init__(self, weights = ROOT / 'yolov5s.pt' , source = 'ROS', conf_thres = 0.4, data_yaml = ROOT / 'data/coco128.yaml' , depth_limit = 2):
        self.weights = weights  # model.pt path(s)
        self.source = str(source)  # file/dir/URL/glob, 0 for webcam
        self.data = data_yaml # path to yaml file
        self.imgsz = (1024, 1024)  # inference size (height, width)
        self.conf_thres = conf_thres  # confidence threshold
        self.iou_thres = 0.65  # NMS IOU threshold
        self.max_det = 1000  # maximum detections per image
        self.device = ''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.view_img = True  # show results
        self.save_txt = False  # save results to *.txt
        self.save_conf = False  # save confidences in --save-txt labels
        self.save_crop = False  # save cropped prediction boxes
        self.nosave = True  # do not save images/videos
        self.classes = None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms = False  # class-agnostic NMS
        self.augment = False  # augmented inference
        self.visualize = False  # visualize features
        self.project = ROOT / 'runs/detect'  # save results to project/name
        self.name = 'exp'  # save results to project/name
        self.exist_ok = False  # existing project/name ok, do not increment
        self.line_thickness = 3  # bounding box thickness (pixels)
        self.half = False  # use FP16 half-precision inference
        self.dnn=False  # use OpenCV DNN for ONNX inference
        self.webcam = True
        self.delta_height = (1080 - 1024)/2
        self.delta_width = (1920 - 1024)/2
        self.depth_limit = depth_limit        
        # This class will start automatically
        self.config()
        self.thread = Thread(target=self.update, args=(), daemon=True)
        self.thread.start()

        # If we have not get our image, we wait
        while True:
            try:
                self.imDisplayed
                break
            except:
                pass

    def config(self):
        # Load model
        self.device = select_device(self.device)
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=self.dnn, data=self.data, fp16=self.half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size

        # Dataloader
        print('[WARNING]: Going into ROS MODE - similar to webcam')
        cudnn.benchmark = True 
        rospy.init_node('YOLOv5', anonymous=False)
        self.dataset = LoadROS(self.source, topic= "/camera/color/rgb/image_raw", fps=30, img_size=self.imgsz, stride=self.stride, auto=self.pt)
        # We should grab the deep channel too!
        #imagedepth_reader = rospy.Subscriber("/camera/depth/rgb/image_rect_raw", Image_ROS, self.imagedepth_callback)
        self.spin_thread = Thread(target=lambda: rospy.spin(), daemon=True)
        self.spin_thread.start()

        self.bs = 1 # batch_size
        self.format = 'webcam'

        # Run inference
        self.model.warmup(imgsz=(1 if self.pt else self.bs, 3, *self.imgsz))  # warmup
        dt, self.seen = [0.0, 0.0, 0.0], 0
        # MOT tracker
        self.mot_tracker = Sort(max_age=60, min_hits=30, iou_threshold=self.iou_thres)
        # Max_age = Maximum number of frames to keep alive a track without associated detections."
        # min_hits = Minimum number of associated detections before track is initialised
        self.objects_characteristics = {'id':[], 'class':[], 'size':[]} # this will be a dictionary. Old school python

        # We also want to start a thread for the size estimation
        self.size_queue = Queue(maxsize=100)
        self.size_threads =[] # We dont know how many will be working at the same time, so we need a counter
        self.size_threads_counter = 0

    def update(self):
        # Get an image constantly from ROS
        while True:
            path, img_raw, imgT, imgRP, vid_cap, s, img_idx = next(self.dataset)
            
            imgRP = torch.from_numpy(imgRP).to(self.device)
            imgRP = imgRP.half() if self.model.fp16 else imgRP.float()  # uint8 to fp16/32
            imgRP /= 255  # 0 - 255 to 0.0 - 1.0
            if len(imgRP.shape) == 3:
                imgRP = imgRP[None]  # expand for batch dim

            # Inference
            pred = self.model(imgRP, augment=self.augment, visualize=False)

            # NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                track_bb_ids =  self.mot_tracker.update([x.cpu().data.numpy() for x in pred][0])
                self.seen += 1
                if self.webcam:  # batch_size >= 1
                    p, im0, frame = path, imgT.copy(), img_idx
                    s += f'{i}: '

                p = Path(p)  # to Path
                annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(imgRP.shape[2:], det[:, :4], im0.shape).round()

                    for c in range(len(track_bb_ids.tolist())): # For each object on SORT
                        # We get details
                        coords = track_bb_ids.tolist()[c]
                        x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
                        name_idx = int(coords[4])
                        class_idx = int(coords[5])
                        name = f'ID:{str(name_idx)}-Class{str(class_idx)}'

                        #if self.get_depth() < self.depth_limit:
                        # And we draw the box around
                        cv2.rectangle(annotator.result(),(x1,y1), (x2,y2), [255,0,0], 2)
                        cv2.putText(annotator.result(), name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, .5, [255,0,0], 2)
                        # We count all objects that pass from right to left
                        if ((x1+x2)/2 < 512):
                            if not (name_idx in self.objects_characteristics['id']):
                                self.objects_characteristics['id'].append(name_idx)
                                self.objects_characteristics['class'].append(class_idx)
                                # And here we should append also the size, but i guess that is thread dependend
                                self.size_queue.put(class_idx)
                                self.size_threads.append(None)
                                self.size_threads[self.size_threads_counter] = Thread(target = self.get_size, 
                                    args=(name_idx,self.size_threads_counter), 
                                    daemon = True)
                                self.size_threads[self.size_threads_counter].start()
                                self.size_threads_counter += 1
 
                cv2.line(annotator.result(),(512,0),(512,1024), color=(255,0,0))
                # Stream results
                im0 = annotator.result()
                if self.view_img:
                    self.imDisplayed = img_raw.copy()
                    self.imDisplayed[int(self.delta_height):int(1024 + self.delta_height), int(self.delta_width):int(1024 + self.delta_width)] = im0

    def get_depth(self):
        return 0.2

    def get_size(self, name_idx, thread_n):
        """
        Size estimator to be run in threads. As we use FIFO and dictionaries, as we are not sure how many objects will be measured,
        we have to wait for the previus thread

        inputs:
            thread_n - Thread number
            ROI - corresponding to coordinates of the box given by SORT
            DEEP_image - Depth image given by ROS topic depth
        
        output:
            fruit box (m) 


        """
        # Wait for the previus thread to finish"
        if thread_n != 0:
            while self.size_threads[thread_n-1].is_alive():
                time.sleep(0.1)
                pass
        # Once that the previus is finish, we can get the next object and append it
        val = self.size_queue.get()
        self.objects_characteristics['size'].append(val)
        self.size_queue.task_done()
        # This should kill the thread
        sys.exit()


    def __iter__(self):
        return self

    def __next__(self):
        return self.imDisplayed