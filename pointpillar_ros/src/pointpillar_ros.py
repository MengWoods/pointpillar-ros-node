#! /usr/bin/env python3
# MIT License

# Copyright (c) 2024 Menghao Woods

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import rospy    # ROS communication
import rospkg   # ROS Package management
from sensor_msgs.msg import PointCloud2 # Message type for point cloud sensor data
import sensor_msgs.point_cloud2 as pc2  # Point cloud data utility function
from std_msgs.msg import Header         # Header in a message
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray  # For detection bounding boxes
import numpy as np
from pyquaternion import Quaternion
# Import needed functions from MMDetection3D library to use neural network model
from mmdet3d.apis import init_model, inference_detector
import time

class PointPillarRos:
    def __init__(self):
        """Initialize the PointPillarRos class."""
        self.initRos()
        self.initPointPillar()

    def initRos(self):
        """Initialize ROS parameters and topics."""
        self.verbose = rospy.get_param('~verbose', False)
        self.model_file_name = rospy.get_param('~model_file_name', 'model_file_name_not_set')
        self.config_file_name = rospy.get_param('~config_file_name', 'config_file_name_not_set')
        self.detection_score_threshold = rospy.get_param('~detection_score_threshold', 0.2)
        self.sub_point_cloud_topic = rospy.get_param('~sub_point_cloud_topic', 'input_point_cloud_topic_not_set')
        self.pub_detection_topic = rospy.get_param('~pub_detection_topic', 'pub_detection_topic_not_set')
        self.sub_point_cloud = rospy.Subscriber(self.sub_point_cloud_topic, PointCloud2, self.lidarCallback, queue_size=1,  buff_size=2**12)
        self.pub_detection = rospy.Publisher(self.pub_detection_topic, BoundingBoxArray, queue_size=1)
        self.pub_point_cloud = rospy.Publisher("/detections", PointCloud2, queue_size=1)
        rospy.loginfo("Pointpillar ros node initialized with the following settings:\n"
                    "\tModel File Name: %s\n"
                    "\tConfig File Name: %s\n"
                    "\tdetection_score_threshold: %s\n"
                    "\tInput Point Cloud Topic: %s\n"
                    "\tOutput Detection Topic: %s",
                    self.model_file_name, self.config_file_name, self.detection_score_threshold,
                    self.sub_point_cloud_topic, self.pub_detection_topic)
    
    def initPointPillar(self):
        """Initialize PointPillar model."""
        self.model = init_model(self.config_file_name, self.model_file_name)
        self.model.cuda() # Load the neural network model into GPU
        rospy.loginfo("PointPillar model is initialized!")
    
    def lidarCallback(self, msg):
        start_time = time.time()  # Start the timer
        # Read point cloud data from msg, skipping NaN values, and extracting defined fields
        pc_data = pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z", "intensity", "ring"))
        # Convert point cloud data to a numpy array
        pc_np = np.array(list(pc_data), dtype=np.float16)
        x = pc_np[:, 0].reshape(-1)
        y = (pc_np[:, 1]).reshape(-1)
        z = (pc_np[:, 2]).reshape(-1)
        # Project the intensity values from 0-255 to 0-1
        i = (pc_np[:, 3] / 255.0).reshape(-1)
        pc_np = np.stack((x, y, z, i)).T
        preprocess_time = time.time() - start_time  # Calculate the time taken for preprocessing

        if hasattr(self, 'model'):
            inference_start_time = time.time()  # Start the timer for inferencing
            # Perform inference using the model on the preprocessed point cloud data
            detections, data = inference_detector(self.model, pc_np)
            inference_time = time.time() - inference_start_time  # Calculate the time taken for inferencing
            self.publishDetection(detections, msg)
            if self.verbose:
                rospy.loginfo("Preprocess Time: %.4f ms | Inference Time: %.4f ms" % (preprocess_time*1000, inference_time*1000))
        else:
            # Log a message if the model is not defined yet
            rospy.loginfo("Waiting for PointPillar model's initialization.")

    def publishDetection(self, detections, msg):
        # Extract bboxes, scores, and lables from the detection
        bounding_boxes = detections.pred_instances_3d.bboxes_3d.tensor
        scores = detections.pred_instances_3d.scores_3d     # Detection confidence score
        labels = detections.pred_instances_3d.labels_3d     # Obstacle category
        # Filter out detections with scores below the threshold
        mask = scores > self.detection_score_threshold
        scores = scores[mask]
        labels = labels[mask]
        bounding_boxes = bounding_boxes[mask]
        # Create a BoundingBoxArray message to store the bounding boxes
        arr_bbox = BoundingBoxArray()
        for i in range(bounding_boxes.shape[0]):
            bbox = BoundingBox()
            bbox.header.frame_id = msg.header.frame_id
            bbox.header.stamp = msg.header.stamp
            # Set position
            bbox.pose.position.x = float(bounding_boxes[i][0])
            bbox.pose.position.y = float(bounding_boxes[i][1])
            bbox.pose.position.z = float(bounding_boxes[i][2]) + float(bounding_boxes[i][5]) / 2
            # Set dimensions
            bbox.dimensions.x = float(bounding_boxes[i][3])  # width
            bbox.dimensions.y = float(bounding_boxes[i][4])  # length
            bbox.dimensions.z = float(bounding_boxes[i][5])  # height
            # Set orientation
            q = Quaternion(axis=(0, 0, 1), radians=float(bounding_boxes[i][6]))
            bbox.pose.orientation.x = q.x
            bbox.pose.orientation.y = q.y
            bbox.pose.orientation.z = q.z
            bbox.pose.orientation.w = q.w
            bbox.value = scores[i]
            bbox.label = labels[i]
            arr_bbox.boxes.append(bbox)
        # Set header for BoundingBoxArray
        arr_bbox.header.frame_id = msg.header.frame_id
        arr_bbox.header.stamp = msg.header.stamp
        self.pub_detection.publish(arr_bbox)

def main():
    # Init pointpillar ROS node with the name
    package_name = "pointpillar-ros-node"
    # Set it to run anonymously for uniqueness.
    rospy.init_node(package_name, anonymous=True)
    # Instantiate the class
    point_pillar = PointPillarRos()
    try:
        rospy.spin()
    except Exception as e:
        del point_pillar
        rospy.logwarn("Launch node failed. Following exception occured: %s" % e)

if __name__ == "__main__":
    main()