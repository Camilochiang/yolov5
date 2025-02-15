#!/home/ubuntu/Agroscope/Python_venvironments/YOLO_v5_venv/bin/python3
from datetime import date, datetime
import cv2 as cv
import os
import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import threading
from queue import Queue

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    print(f'Adding: {ROOT} to sys path')
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend

from utils.datasets_agroscope import IMG_FORMATS, VID_FORMATS, LoadROS
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

# Tracker
from trackers.ocsort.ocsort import OCSort

@torch.no_grad()


class YOLOv5():
    def __init__(self, dataset , weights = ROOT / 'yolov5s.pt' , source = 'ROS', conf_thres = 0.4, data_yaml = ROOT / 'data/coco128.yaml', tracker_sense = True):
        print(colorstr('[ASPEN - DETECTOR]: Starting Yolo_v5 wrapper'))
        self.dataset = dataset
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
        self.FPS = 15

        self.queue_frame = Queue()
        self.queue_stand_by_frame = Queue(maxsize = 1)
        self.seen =  -1 
        self.update_running = True

        # Tracker
        self.mot_tracker = OCSort(det_thresh = self.conf_thres, max_age = 10, min_hits = 1, iou_threshold = 0.01, use_byte = False) # OCSort sort
        self.tracker_limit = int(self.imgsz[0]*0.75) if tracker_sense == True else int(self.imgsz[0]*.25)
        self.tracked_detections = []
        self.max_count_MOT = 0


        # Data to be published and used by ROS
        self.newest_detections = [] # A list of list that contain coordinates, category, ID and score for each detection no previusly published
        self.queue_detections = Queue()

        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        # Load model
        self.device = select_device(self.device)
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=self.dnn, data=self.data, fp16=self.half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size

        # Dataloader
        cudnn.benchmark = True 
        self.bs = 1 # batch_size
        self.format = 'webcam'

        # Run inference
        self.model.warmup(imgsz=(1 if self.pt else self.bs, 3, *self.imgsz))  # warmup
        
        for status, path, img_raw, imgT, imgRP, vid_cap, s, img_idx in self.dataset:
            if self.update_running:
                if status and self.seen != img_idx: # Not sure why but working with threads and queue, the queue gave the same object several times. This is a protection to avoid re-analyse the same object
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
                        # Tracker update
                        self.online_targets = self.mot_tracker.update(det[:,0:6].cpu().numpy()) # OCSORT

                        self.im0 = imgT.copy()
                        annotator = Annotator(self.im0, line_width=self.line_thickness, example=str(self.names))

                        if len(self.online_targets):
                            self.newest_detections = []

                            for tracked_detection in self.online_targets:
                                # tracked detections give the following array [x1,y1,x2,y2,track_ID, probability, category]
                                annotator.box_label(tracked_detection[0:4], None, color=colors(3, True))
                                # We count all objects that pass from either right to left or vise versa
                                if (tracked_detection[0]+tracked_detection[2])/2 > self.tracker_limit:                                    
                                    if not tracked_detection[4] in self.tracked_detections:
                                        # We have to re add the borders to have in proper camera coordinates!
                                        tracked_detection[0:4] = [tracked_detection[0] + self.delta_width,
                                                                  tracked_detection[1] + self.delta_height,
                                                                  tracked_detection[2] + self.delta_width,
                                                                  tracked_detection[3] + self.delta_height]

                                        self.tracked_detections.append(tracked_detection[4])
                                        self.max_count_MOT += 1
                                        self.newest_detections.append(tracked_detection)
                        #_ = cv.line(annotator.result(), (self.tracker_limit,0),(self.tracker_limit,self.imgsz[0]), color=(255,0,0))

                        self.im0 = annotator.result()
                        self.queue_frame.put(self.im0)
                        self.queue_detections.put(self.newest_detections)

                        # We will copy the image to queue_stand_by_frame so it doesnt block the iteration
                        self.queue_stand_by_frame.put(self.im0)

                        self.seen += 1
                else:
                    self.queue_stand_by_frame.put(img_raw)
            else:
                break                        

    def close(self):
        self.update_running = False
        print(colorstr('[ASPEN - DETECTOR]: We count '+ str(self.max_count_MOT) +' objects'))
        print(colorstr('[ASPEN - DETECTOR]: Closing Yolo_v5 wrapper'))


    def __iter__(self):
        return self


    def __next__(self):
        if self.queue_frame.qsize() > 0:
            return True, self.queue_frame.get(),self.queue_detections.get(), self.seen
        else:
            return False, self.queue_stand_by_frame.get(),False, self.seen