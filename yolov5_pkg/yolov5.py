import rclpy
from rclpy.node import Node

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from cv_bridge import CvBridge
from pathlib import Path
import os
import sys

from sensor_msgs.msg import Image
from yolov5_interface.msg import BoundingBox, BoundingBoxes

# add yolov5 submodule to path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] / 'yolov5'
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative path

# import from yolov5 submodules
from models.common import DetectMultiBackend
from utils.general import (
    check_img_size,
    non_max_suppression,
    scale_boxes
)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
from utils.augmentations import letterbox

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Yolov5(Node):
    def __init__(self):
        super().__init__("yolov5")
        self.conf_thres = self.get_parameter("confidence_threshold").value
        self.iou_thres = self.get_parameter("iou_threshold")
        self.agnostic_nms = self.get_parameter("agnostic_nms")
        self.max_det = self.get_parameter("maximum_detections")
        self.classes = self.get_parameter("classes", None)
        self.view_image = self.get_parameter("view_image")
        self.line_thickness = self.get_parameter("line_thickness")
        # Initialize weights 
        weights = self.get_parameter("weights")
        # Initialize model
        self.device = torch.device(self.get_parameter("device"))
        self.optimize_mode = self.get_parameter("optimize")
        self.model = DetectMultiBackend(
            weights, 
            device=self.device, 
            dnn=self.get_parameter("dnn"), 
            data=self.get_parameter("data")
        )
        self.stride, self.names, self.pt, self.jit, self.onnx, self.engine = (
            self.model.stride,
            self.model.names,
            self.model.pt,
            self.model.jit,
            self.model.onnx,
            self.model.engine
        )
        
        self.img_size = [self.get_parameter("inference_size_w"), self.get_parameter("inference_size_h")]
        self.img_size = check_img_size(self.img_size, s=self.stride)
        
        cudnn.benchmark = True  # set True to speed up constant image size inference
        self.model.warmup()  # warmup
        self.input_topic_name = self.get_parameter('input_image_topic')
        self.output_topic_name = self.get_parameter('output_topic')
        
        self.subscription = self.create_subscription(
            Image, 
            'video_frames', 
            self.inference, 
            10
        )
        self.publisher_ = self.create_publisher(BoundingBoxes, 'yolov5_bboxes', 10)
        self.bridge = CvBridge()
        
    def inference(self, data):
        self.get_logger().info('Receiving video frame')
        current_frame = self.bridge.imgmsg_to_cv2(data)
        im, im0 = self.preprocess(current_frame)
        im = torch.from_numpy(im).to(self.device) 
        im /= 255
        if len(im.shape) == 3:
            im = im[None]
        pred = self.model(im, augment=False, visualize=False)
        pred = non_max_suppression(
            pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det
        )
        det = pred[0].cpu().numpy()

        bounding_boxes = BoundingBoxes()
        bounding_boxes.header = data.header
        bounding_boxes.image_header = data.header
        annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                bounding_box = BoundingBox()
                c = int(cls)
                # Fill in bounding box message
                bounding_box.Class = self.names[c]
                bounding_box.probability = conf 
                bounding_box.xmin = int(xyxy[0])
                bounding_box.ymin = int(xyxy[1])
                bounding_box.xmax = int(xyxy[2])
                bounding_box.ymax = int(xyxy[3])
                bounding_boxes.bounding_boxes.append(bounding_box)

                # Annotate the image
                if self.publish_image or self.view_image:  # Add bbox to image
                      # integer class
                    label = f"{self.names[c]} {conf:.2f}"
                    annotator.box_label(xyxy, label, color=colors(c, True))       
                ### POPULATE THE DETECTION MESSAGE HERE
            # Stream results
            im0 = annotator.result()
            
        self.publisher_.publish(bounding_boxes)
        self.get_logger().info(f'Publishing Bounding Boxes {len(det)}')
        if self.view_image:
            cv2.imshow('frame', im0)
            cv2.waitKey(1)
    
    def preprocess(self, img):
        img0 = img.copy()
        img = np.array([letterbox(img, self.img_size, stride=self.stride, auto=self.pt)[0]])
        # Convert
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img)

        return img, img0 

def main(args=None):
    rclpy.init(args=args)
    yolov5_node = Yolov5()
    rclpy.spin(yolov5_node)
    yolov5_node.destroy_node()
    rclpy.shutdown()
  
if __name__ == '__main__':
    main()