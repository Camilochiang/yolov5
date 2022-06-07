#!/home/ubuntu/Agroscope/Python_venvironments/YOLO_v5_venv/bin/python3
from datetime import date, datetime
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
from trackers.sort.sort import * # as python code. directly from github
#from ByteTrack.yolox.tracker.byte_tracker import BYTETracker as ByteTracker # The original bytetracker
#from bytetrack_realtime.bytetrack_realtime.byte_tracker import ByteTracker # A realtime implementation
#from trackers.ocsort.ocsort import OCSort 

@torch.no_grad()


class YOLOv5():
    def __init__(self, dataset , weights = ROOT / 'yolov5s.pt' , source = 'ROS', conf_thres = 0.4, data_yaml = ROOT / 'data/coco128.yaml'):
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
        
        # Tracker
        # Max_age = Maximum number of frames to keep alive a track without associated detections."
        # min_hits = Minimum number of associated detections before track is initialised

        # SORT:
        self.mot_tracker = Sort(max_age=10, min_hits=3, iou_threshold=self.iou_thres-0.2)
        # BYTETracker
        #self.mot_tracker = ByteTracker(track_thresh = self.conf_thres, track_buffer = 100, match_thresh = 0, frame_rate = self.FPS)
        # OCSort
        #self.mot_tracker = OCSort(det_thresh = self.conf_thres, max_age = 10, min_hits = 2, iou_threshold = self.iou_thres - 0.1, use_byte = False)

        # Get an image from ROS
        
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
                        #online_targets =  self.mot_tracker.update([x.cpu().data.numpy() for x in pred][0]) # SORT
                        #online_targets = self.mot_tracker.update(det, (1024,1024), (1024,1024))
                        #online_targets = self.mot_tracker.update(det[:,0:5].cpu().numpy()) # OCSORT

                        self.im0 = imgT.copy()

                        annotator = Annotator(self.im0, line_width=self.line_thickness, example=str(self.names))

                        #tracked = self.mot_tracker.get_tracks(2)
                        #for track in tracked:
                        #    annotator.box_label(track[1:5], str(track[0]), color=colors(2, True))

                        if len(det):
                            for *xyxy, conf, cls in reversed(det):
                                annotator.box_label(xyxy, None, color=colors(1, True))

                            #for tracked_object in online_targets:
                            #    annotator.box_label(tracked_object, None, color=colors(3, True))

                            # And we draw the box around
                            #    _ = cv2.rectangle(annotator.result(),(x1,y1), (x2,y2), [255,255,255], 2)
                            #    _ = cv2.putText(annotator.result(), name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, .5, [255,0,0], 2)
                            #     # We count all objects that pass from right to left
                            #     #if ((x1+x2)/2 < 512):
                            #         #pass
                            #         # if not (name_idx in self.objects_characteristics['id']):
                            #         #     self.objects_characteristics['id'].append(name_idx)
                            #         #     self.objects_characteristics['class'].append(class_idx)
                            #         #     # And here we should append also the size, but i guess that is thread dependend
                            #         #     self.size_queue.put(class_idx)
                            #         #     self.size_threads.append(None)
                            #         #     self.size_threads[self.size_threads_counter] = Thread(target = self.get_size, 
                            #         #         args=(name_idx,self.size_threads_counter), 
                            #         #         daemon = True)
                            #         #     self.size_threads[self.size_threads_counter].start()
                            #         #     self.size_threads_counter += 1

                        #cv2.line(annotator.result(),(512,0),(512,1024), color=(255,0,0))
                        # Stream results
                        self.im0 = annotator.result()
                        self.queue_frame.put(self.im0)
                        # We will copy the image to queue_stand_by_frame so it doesnt block the iteration
                        self.queue_stand_by_frame.put(self.im0)

                        self.seen += 1
                else:
                    self.queue_stand_by_frame.put(img_raw)
            else:
                break                        

    def close(self):
        self.update_running = False
        print(colorstr('[ASPEN - DETECTOR]: Closing Yolo_v5 wrapper'))
        sys.exit()


    def __iter__(self):
        return self


    def __next__(self):
        if self.queue_frame.qsize() > 0:
            return True, self.queue_frame.get(), self.seen
        else:
            return False, self.queue_stand_by_frame.get(), self.seen
        