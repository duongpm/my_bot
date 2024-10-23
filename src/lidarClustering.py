#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import numpy as np
import math
from sklearn.cluster import DBSCAN
from matplotlib import pyplot
from numpy import unique
from numpy import where

class ObjectAvoidanceNode(Node):
    def __init__(self):
        super().__init__('object_avoidance_node')
        self.subscription = self.create_subscription(
            LaserScan,
            'scan',
            self.lidar_callback,
            10)
        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.safe_distance = 1.0  # Meters
        self.get_logger().info('Object Avoidance Node Started')

    def lidar_callback(self, msg):
        ranges = msg.ranges
        lidarXY = np.empty([2, 2])
        # convert lidar data  to Cartesian coordinates
        for i,range in enumerate(ranges):
            if (range < msg.range_min or range > msg.range_max):
                x = np.inf
                y = np.inf
            else:
                angle = msg.angle_min + i*msg.angle_increment
                x = range*math.cos(angle)
                y = range*math.sin(angle)
            lidarXY = np.append(lidarXY, [[x,y]],axis=0)

        # define the model
        model = DBSCAN(eps=0.30, min_samples=9)
        # fit model and predict clusters
        yhat = model.fit_predict(lidarXY)
        # retrieve unique clusters
        clusters = unique(yhat)
        # create scatter plot for samples from each cluster
        for cluster in clusters:
            # get row indexes for samples with this cluster
            row_ix = where(yhat == cluster)
            # create scatter of these samples
            pyplot.scatter(lidarXY[row_ix, 0], lidarXY[row_ix, 1])
        # show the plot
        pyplot.show()

        min_distance = min(ranges)
        print(min_distance)
       # print("haha")

        twist_msg = Twist()

        if min_distance < self.safe_distance:
            # Obstacle detected, turn the robot
            twist_msg.linear.x = 0.0
            twist_msg.angular.z = 0.5  # Rotate counter-clockwise
        else:
            # No obstacle detected, move forward
            twist_msg.linear.x = 0.2
            twist_msg.angular.z = 0.0

        self.publisher.publish(twist_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ObjectAvoidanceNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()