"""
This is a simple detection based on yolov5
Input: a string which is encoded from a picture
OutPut: a json which contains classes, bounding boxes and confidence

"""
import time
import argparse
import os
import sys
import numpy as np
from pathlib import Path
from io import BytesIO
import base64
import re
import json
from unittest import result
from numpyencoder import NumpyEncoder
from PIL import Image


import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from utils.augmentations import letterbox

weight_path = "weights/yolov5n.pt"

def model_name() -> str:
    m = re.search("/(.*)\.pt", weight_path)
    if m:
        return m.group(1)
    return ""

class model():
    def __init__(self, weight_path, device="cpu"):
        self.imgsz=(640, 640)  # inference size (height, width)
        bs = 1  # batch size

        self.device = select_device(device) 
        self.model = DetectMultiBackend(weight_path, device=self.device, dnn=False, data='data/coco128.yaml', fp16=False)
        # dnn: use OpenCV DNN for ONNX inference
        #fp16:use FP16 half-precision inference
        
        stride, names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(self.imgsz, s=stride)  # check image size
    def infer(self, im):
        bs = 1
        self.model.warmup(imgsz=(1 if self.pt else bs, 3, *self.imgsz))  # warmup
        seen, windows = 0, []

        im = Image.open(im)
        im = np.asarray(im)
        original_im = im 
        original_shape = original_im.shape
        im = letterbox(im, 640, stride=32, auto=True)[0]
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)
          
        # im = im.reshape(3, im.shape[0],im.shape[1])
        
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        # Inference
        
        pred = self.model(im, augment=False, visualize=False) #augmented inference and visualize features


        # NMS
        pred = non_max_suppression(pred, conf_thres=0.3, iou_thres=0.45, classes=None, agnostic=False, max_det=1000) 
        #agnostic: class-agnostic NMS
        #max_det: # maximum detections per image
        for i, det in enumerate(pred):
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], original_shape).round()
            det[:, :4] = det[:, :4].int()
            det = det.numpy()
            det[:, [4, 5]] = det[:, [5, 4]]
        # change base 0 to base 1

        return det.tolist()
        


def predict(im, weight_path: str) -> list:
    weights = weight_path
    device = 'cpu'# device: cuda device, i.e. 0 or 0,1,2,3 or cpu
    imgsz=(640, 640)  # inference size (height, width)
    bs = 1  # batch size

    device = select_device(device) 
    model = DetectMultiBackend(weights, device=device, dnn=False, data='data/coco128.yaml', fp16=False)
    # dnn: use OpenCV DNN for ONNX inference
    #fp16:use FP16 half-precision inference
    
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Run inference
    bs = 1
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows = 0, []

    im = Image.open(im)
    im = np.asarray(im)
    original_im = im 
    original_shape = original_im.shape
    im = letterbox(im, 640, stride=32, auto=True)[0]
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)
      
    # im = im.reshape(3, im.shape[0],im.shape[1])
    
    im = torch.from_numpy(im).to(device)
    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    # Inference
    
    pred = model(im, augment=False, visualize=False) #augmented inference and visualize features


    # NMS
    pred = non_max_suppression(pred, conf_thres=0.3, iou_thres=0.45, classes=None, agnostic=False, max_det=1000) 
    #agnostic: class-agnostic NMS
    #max_det: # maximum detections per image
    for i, det in enumerate(pred):
        det[:, :4] = scale_coords(im.shape[2:], det[:, :4], original_shape).round()
        det[:, :4] = det[:, :4].int()
        det = det.numpy()
        det[:, [4, 5]] = det[:, [5, 4]]

    return det.tolist()


def image_str2file(image_str):
    im_bytes = base64.b64decode(image_str)
    im_file = BytesIO(im_bytes)
    return im_file

def image_to_box(image_str, weight_path: str) -> list:
    im_file = image_str2file(image_str)
    
    result = predict(im_file, weight_path)
    # out_scores, out_boxes, out_classes = predict(im_file)
    # assert len(out_scores) == len(out_boxes) == len(out_classes)
    # def add(n):
    #     return n + 1
    # out_classes = list(map(add, out_classes))
    # # change coco classes starting from index 0 to index 1
    # # for example, coco car class: 2 -> 3
    # result = {
    #     "scores": out_scores,
    #     "boxes": out_boxes,
    #     "classes": out_classes,
    #     "len": len(out_boxes)
    # }
    # result = json.dumps(result, cls=NumpyEncoder)
    return result

def test():
    with open("data/images/1.jpg", "rb") as imagestring:
        convert_string = base64.b64encode(imagestring.read())
    result = image_to_box(convert_string, weight_path)
    print("model name is " + model_name())
    yolo = model('weights/yolov5x.pt')
    start = time.perf_counter()
    result = yolo.infer("data/images/1.jpg")
    end = time.perf_counter()
    print(end - start)
    return result


if __name__ == "__main__":
    print(test())