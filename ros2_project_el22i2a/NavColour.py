
import threading
import sys, time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from rclpy.exceptions import ROSInterruptException
import signal
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from math import sin, cos
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String




class ColourDetector(Node):
    def __init__(self):
        super().__init__('robot')
     
        self.colour_pub = self.create_publisher(String, 'detected_colour', 10)
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(Image, '/camera/image_raw', self.callback, 10)
        self.subscription 
        self.sensitivity = 10
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.rate = self.create_rate(10)  


    def callback(self, data):
        image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        hsv_green_lower = np.array([60 - self.sensitivity, 100, 100])
        hsv_green_upper = np.array([60 + self.sensitivity, 255, 255])
        hsv_red_lower1 = np.array([0, 100, 100])
        hsv_red_upper1 = np.array([10, 255, 255])
        hsv_red_lower2 = np.array([170, 100, 100])
        hsv_red_upper2 = np.array([180, 255, 255])
        hsv_blue_lower = np.array([100, 150, 50])
        hsv_blue_upper = np.array([140, 255, 255])

        masks = {
            'Green': (cv2.inRange(hsv, hsv_green_lower, hsv_green_upper), (0, 255, 0)),
            'Red': (cv2.bitwise_or(
                        cv2.inRange(hsv, hsv_red_lower1, hsv_red_upper1),
                        cv2.inRange(hsv, hsv_red_lower2, hsv_red_upper2)
                    ), (0, 0, 255)),
            'Blue': (cv2.inRange(hsv, hsv_blue_lower, hsv_blue_upper), (255, 0, 0))
        }


        for colour_name, (mask, colour_bgr) in masks.items():
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest)

                if area > 100:
                    cv2.drawContours(image, [largest], -1, colour_bgr, 2)
                    M = cv2.moments(largest)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cv2.putText(image, f"{colour_name}", (cx - 30, cy - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour_bgr, 2)
                        print(f"{colour_name} detected at: ({cx}, {cy}) with area: {area:.1f}")
                        if colour_name == 'Blue':
                            msg = String()
                            msg.data = 'Blue'
                            self.colour_pub.publish(msg)
                            

        cv2.namedWindow('Detected Colours', cv2.WINDOW_NORMAL)
        cv2.imshow('Detected Colours', image)
        cv2.resizeWindow('Detected Colours', 320, 240)
        cv2.waitKey(3)



class NavigationController(Node):
    def __init__(self):
        super().__init__('navigation_controller')

        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.goal_in_progress = False
        self.current_goal_index = 0
        self.subscription = self.create_subscription(String, 'detected_colour', self.colour_callback, 10)
        self.goals = [
            (-2.0, -5.0, 0.0),
            (0.0, -8.0, 3.14)
        ]

        threading.Thread(target=self.navigation_loop, daemon=True).start()

    def navigation_loop(self):
        self.nav_client.wait_for_server()
        while rclpy.ok() and self.current_goal_index < len(self.goals):
            if not self.goal_in_progress:
                x, y, yaw = self.goals[self.current_goal_index]
                self.send_goal(x, y, yaw)

    def send_goal(self, x, y, yaw):
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.position.z = 0.0
        goal_msg.pose.pose.orientation.z = sin(yaw / 2)
        goal_msg.pose.pose.orientation.w = cos(yaw / 2)

        self.goal_in_progress = True
        send_goal_future = self.nav_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.goal_in_progress = False
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def result_callback(self, future):
        result = future.result().result
        self.current_goal_index += 1
        self.goal_in_progress = False

    def colour_callback(self, msg):
        if msg.data == 'Blue' and not self.goal_in_progress:
            self.get_logger().info("Blue detected! Navigating to target position...")
            self.send_goal(-3.5, -11.0, 1.55)


    

def main():
    rclpy.init()

    nav_controller = NavigationController()
    colour_detector = ColourDetector()

    executor = MultiThreadedExecutor()
    executor.add_node(colour_detector)
    executor.add_node(nav_controller)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        colour_detector.destroy_node()
        nav_controller.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
