import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import numpy as np

class ObjectFollowerNode(Node):
    def __init__(self):
        super().__init__('object_follower_node')
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.get_logger().info('Object Follower Node Initialized')
        
    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        lower_red = np.array([0, 120, 70])
        upper_red = np.array([10, 255, 255])
        mask = cv2.inRange(hsv, lower_red, upper_red)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.circle(cv_image, (center_x, center_y), 5, (0, 0, 255), -1)
            self.follow_object(center_x, center_y, cv_image.shape[1], cv_image.shape[0])
        cv2.imshow('Object Following', cv_image)
        cv2.waitKey(1)

    def follow_object(self, center_x, center_y, image_width, image_height):
        cmd_vel = Twist()
        error_x = center_x - image_width // 2
        error_y = center_y - image_height // 2
        if error_x > 100:
            cmd_vel.angular.z = -0.1 
        elif error_x < -100:
            cmd_vel.angular.z = 0.1
        else:
            cmd_vel.angular.z = 0.0  
        
        if error_y > 100:
            cmd_vel.linear.x = 0.2 
        elif error_y < -100:
            cmd_vel.linear.x = -0.2  
        else:
            cmd_vel.linear.x = 0.0 
        self.cmd_vel_pub.publish(cmd_vel)

def main(args=None):
    rclpy.init(args=args)
    object_follower = ObjectFollowerNode()
    rclpy.spin(object_follower)
    object_follower.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
