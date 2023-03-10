import rclpy
from rclpy.node import Node

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from cv_bridge import CvBridge
import os
import sys

from sensor_msgs.msg import Image
from yolov5_interfaces.msg import BoundingBox, BoundingBoxes

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# add yolov5 submodule to path
WORKDIR_PATH = os.environ['ROS_WS']
sys.path.append(os.path.join(WORKDIR_PATH, 'src/yolov5_pkg/yolov5_pkg/yolov5'))

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
        self.declare_parameter("confidence_threshold", 0.5)
        self.conf_thres = self.get_parameter("confidence_threshold")._value
        self.declare_parameter("iou_threshold", 0.45)
        self.iou_thres = self.get_parameter("iou_threshold")._value
        self.declare_parameter("agnostic_nms", True)
        self.agnostic_nms = self.get_parameter("agnostic_nms")._value
        self.declare_parameter("maximum_detections", 1000)
        self.max_det = self.get_parameter("maximum_detections")._value
        self.classes = None
        self.declare_parameter("view_image", False)
        self.view_image = self.get_parameter("view_image")._value
        self.declare_parameter("line_thickness", 3)
        self.line_thickness = self.get_parameter("line_thickness")._value
        # Initialize weights 
        self.declare_parameter("weights", 'yolov5m.pt')
        weights = self.get_parameter("weights")._value
        # Initialize model
        self.declare_parameter("device", 'cuda')
        self.device = torch.device(self.get_parameter("device")._value)
        self.declare_parameter("optimize", 'vanilla')
        self.optimize_mode = self.get_parameter("optimize")
        self.declare_parameter("dnn", True)
        self.declare_parameter("data", 'coco128.yaml')
        self.model = DetectMultiBackend(
            weights, 
            device=self.device, 
            dnn=self.get_parameter("dnn")._value, 
            data=self.get_parameter("data")._value
        )
        self.stride, self.names, self.pt, self.jit, self.onnx, self.engine = (
            self.model.stride,
            self.model.names,
            self.model.pt,
            self.model.jit,
            self.model.onnx,
            self.model.engine
        )
        self.declare_parameter("inference_size_w", 640)
        self.declare_parameter("inference_size_h", 640)
        self.img_size = [self.get_parameter("inference_size_w")._value, self.get_parameter("inference_size_h")._value]
        self.img_size = check_img_size(self.img_size, s=self.stride)
        
        cudnn.benchmark = True  # set True to speed up constant image size inference
        self.model.warmup()  # warmup
        self.declare_parameter("input_image_topic", 'video_frames')
        self.declare_parameter("output_topic", 'yolo_res')
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
        im = torch.from_numpy(im).to(self.device).float()
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
                bounding_box.class_id = self.names[c]
                bounding_box.probability = float(conf) 
                bounding_box.xmin = int(xyxy[0])
                bounding_box.ymin = int(xyxy[1])
                bounding_box.xmax = int(xyxy[2])
                bounding_box.ymax = int(xyxy[3])
                bounding_boxes.bounding_boxes.append(bounding_box)

                # Annotate the image
                if self.view_image:  # Add bbox to image
                    label = f"{self.names[c]} {conf:.2f}"
                    annotator.box_label(xyxy, label, color=colors(c, True))       
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