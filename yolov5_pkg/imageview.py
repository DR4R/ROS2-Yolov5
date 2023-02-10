import rclpy
from rclpy.node import Node 
from sensor_msgs.msg import Image 
import cv2
from message_filters import ApproximateTimeSynchronizer, Subscriber
import numpy as np

from yolov5_interfaces.msg import BoundingBoxes

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.image_subscription = Subscriber(
            self,
            Image,
            'video_frames'
        )
        self.bbox_subscription = Subscriber(
            self,
            BoundingBoxes,
            'yolov5_bboxes'
        )
        self.timsSync = ApproximateTimeSynchronizer(
            [
                self.image_subscription,
                self.bbox_subscription
            ],
            30,
            0.01
        )
        self.timsSync.registerCallback(self.callback)
   
    def callback(self, img: Image, bboxes):
        self.get_logger().info('Receiving video frame')
        h, w = img.height, img.width
        img = np.array(img.data, dtype=np.uint8, copy=True)
        img = np.reshape(img, (h, w, 3))
        for bbox in bboxes.bounding_boxes:
            cv2.putText(img, bbox.class_id, (bbox.xmin, bbox.ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36,255,12), 2)
            cv2.putText(img, f"{bbox.probability:.3f}", (bbox.xmin + 90, bbox.ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36,255,12), 2)
            cv2.rectangle(img, (bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax), (255,0,0), 2)
        cv2.imshow("camera", img)
        cv2.waitKey(1)
        
  
def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)
    image_subscriber.destroy_node()
    rclpy.shutdown()
  
if __name__ == '__main__':
    main()