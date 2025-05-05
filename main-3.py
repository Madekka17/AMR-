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
        # Convert the ROS image message to a CV2 image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # Convert the image to HSV for color-based detection
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        
        # Define the range for the color (in this case, tracking a red object)
        lower_red = np.array([0, 120, 70])
        upper_red = np.array([10, 255, 255])
        
        # Mask the image to get only the red objects
        mask = cv2.inRange(hsv, lower_red, upper_red)
        
        # Find contours in the masked image
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get the bounding box around the largest contour
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Draw the rectangle on the original image
            cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Calculate the center of the object
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Draw the center point
            cv2.circle(cv_image, (center_x, center_y), 5, (0, 0, 255), -1)
            
            # Control the robot based on the object's position
            self.follow_object(center_x, center_y, cv_image.shape[1], cv_image.shape[0])
        
        # Show the image for debugging purposes
        cv2.imshow('Object Following', cv_image)
        cv2.waitKey(1)

    def follow_object(self, center_x, center_y, image_width, image_height):
        # Create the Twist message to send to the robot
        cmd_vel = Twist()
        
        # Calculate the error in x and y coordinates
        error_x = center_x - image_width // 2
        error_y = center_y - image_height // 2
        
        # If the object is too far to the left, turn left
        if error_x > 100:
            cmd_vel.angular.z = -0.1  # Turn right
        # If the object is too far to the right, turn right
        elif error_x < -100:
            cmd_vel.angular.z = 0.1  # Turn left
        else:
            cmd_vel.angular.z = 0.0  # No turning
        
        # If the object is too high, move forward
        if error_y > 100:
            cmd_vel.linear.x = 0.2  # Move forward
        # If the object is too low, move backward
        elif error_y < -100:
            cmd_vel.linear.x = -0.2  # Move backward
        else:
            cmd_vel.linear.x = 0.0  # Stop moving forward/backward
        
        # Publish the velocity command to move the robot
        self.cmd_vel_pub.publish(cmd_vel)

def main(args=None):
    rclpy.init(args=args)
    object_follower = ObjectFollowerNode()
    rclpy.spin(object_follower)
    
    # Destroy the node when exiting
    object_follower.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
