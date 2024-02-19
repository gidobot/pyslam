import rospy
from geometry_msgs.msg import TransformStamped, PoseStamped, Pose, Point, Quaternion
from std_msgs.msg import Header
import tf2_geometry_msgs
import tf2_ros

import numpy as np
from os import path as osp
import csv
from transforms3d.quaternions import mat2quat, qconjugate, qmult, quat2mat

root_dir = "/media/Data/PSTAR/Reconstruction_Datasets/UWHandles/20181221/SlopeAndMagneticSeep/set1/arrangement3/tagslam"
input_gt = None
with open(osp.join(root_dir, "camera_poses.txt")) as file:
    input_gt = np.array(list(csv.reader(file)), np.double)
input_tracker = None
with open(osp.join(root_dir, "tracker_poses_SIFT.txt")) as file:
    input_tracker = np.array(list(csv.reader(file)), np.double)
seq_length = input_gt.shape[0]
gt_idxs = input_tracker[:,0]

# tf_buffer = tf2_ros.Buffer(rospy.Duration(1200.0)) # tf buffer length
# tf_listener = tf2_ros.TransformListener(self.tf_buffer)
tf_br = tf2_ros.TransformBroadcaster()

def read_poses(idx):
    row_tracker = input_tracker[idx]
    pose_diff = row_tracker[3:10]
    row_gt = input_gt[int(gt_idxs[idx])]
    pose_fish = row_gt[1:8]
    pose_left = row_gt[8:15]
    return pose_diff, pose_left, pose_fish

def publish_tfs(idx):
    tf_map_left = TransformStamped()
    tf_map_fish = TransformStamped()
    tf_left_fish = TransformStamped()

    tf_map_left.header.frame_id = "map"
    tf_map_fish.header.frame_id = "map"
    tf_left_fish.header.frame_id = "left" 

    tf_map_left.child_frame_id = "left"
    tf_map_fish.child_frame_id = "fish"
    tf_left_fish.child_frame_id = "fish_est"

    currTime = rospy.Time.now()
    tf_map_left.header.stamp = currTime
    tf_map_fish.header.stamp = currTime
    tf_left_fish.header.stamp = currTime

    pose_diff, pose_left, pose_fish = read_poses(idx)

    # invert for proper tf structure
    T_left = np.eye(4)
    T_left[:3,3] = pose_left[:3]
    T_left[:3,:3] = quat2mat(pose_left[3:7])
    T_left_inv = np.linalg.inv(T_left)
    q_left = mat2quat(T_left_inv[:3,:3])
    t_left = T_left_inv[:3,3]
    tf_map_left.transform.translation = Point(*t_left)
    tf_map_left.transform.rotation = Quaternion(*q_left[[1,2,3,0]])

    T_fish = np.eye(4)
    T_fish[:3,3] = pose_fish[:3]
    T_fish[:3,:3] = quat2mat(pose_fish[3:7])
    T_fish_inv = np.linalg.inv(T_fish)
    q_fish = mat2quat(T_fish_inv[:3,:3])
    t_fish = T_fish_inv[:3,3]
    tf_map_fish.transform.translation = Point(*t_fish)
    tf_map_fish.transform.rotation = Quaternion(*q_fish[[1,2,3,0]])

    T_diff_gt = T_left.dot(T_fish_inv)
    t_diff_gt = T_diff_gt[:3,3]
    t_diff = pose_diff[:3]*np.linalg.norm(t_diff_gt)/np.linalg.norm(pose_diff[:3])
    # t_diff = pose_diff[:3]*np.linalg.norm(t_diff_gt)
    tf_left_fish.transform.translation = Point(*t_diff)
    tf_left_fish.transform.rotation = Quaternion(*pose_diff[3:7][[1,2,3,0]])

    tf_br.sendTransform([tf_map_left, tf_map_fish, tf_left_fish])

if __name__ == '__main__':
    rospy.init_node('ros_visualize')

    rate = rospy.Rate(3) 
    idx = 0
    while idx < seq_length and not rospy.is_shutdown():
        print(idx)
        publish_tfs(idx)
        idx += 1
        rate.sleep()