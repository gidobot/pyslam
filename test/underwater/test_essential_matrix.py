import sys 
import numpy as np
import cv2 
from matplotlib import pyplot as plt
from transforms3d.quaternions import mat2quat, qconjugate, qmult, quat2mat
import csv
from os import path as osp

sys.path.append("../../")
from config import Config

from mplot_figure import MPlotFigure
from feature_tracker import feature_tracker_factory, FeatureTrackerTypes 
from feature_manager import feature_manager_factory
from feature_types import FeatureDetectorTypes, FeatureDescriptorTypes, FeatureInfo
from feature_matcher import feature_matcher_factory, FeatureMatcherTypes
from utils_img import combine_images_horizontally, rotate_img, transform_img, add_background
from utils_geom import add_ones
from utils_features import descriptor_sigma_mad, compute_hom_reprojection_error
from utils_draw import draw_feature_matches
from utils_knfu_slam import normalizeFisheye, normalizePerspective

from feature_tracker_configs import FeatureTrackerConfigs

from timer import TimerFps

#============================================
# Helper Functions  
#============================================  

uw_root_dir = "/media/Data/PSTAR/Reconstruction_Datasets/UWHandles/20181221/SlopeAndMagneticSeep/set1/arrangement3/tagslam"
def read_pose(idx):
    # gt tf Translation: [-0.089, -0.364, 1.328]
    #       Rotation: in Quaternion [-0.305, 0.061, 0.149, 0.939]
    with open(osp.join(uw_root_dir, "camera_poses.txt")) as file:
        reader = csv.reader(file)
        row = np.array(list(reader)[idx], np.double)
        t_fish = row[1:4]
        q_fish = row[4:8] # wxyz
        T_fish = np.eye(4)
        T_fish[:3,:3] = quat2mat(q_fish)
        T_fish[:3,3] = t_fish
        T_fish = np.linalg.inv(T_fish)
        # invert stereo pose to get world to stereo tf
        t_left = row[8:11]
        q_left = row[11:15]
        T_left = np.eye(4)
        T_left[:3,:3] = quat2mat(q_left)
        T_left[:3,3] = t_left
        # T_left = np.linalg.inv(T_left)
        # take tf difference between stereo and fisheye
        T_diff = T_left.dot(T_fish)
        t_diff = T_diff[:3,3]
        q_diff = mat2quat(T_diff[:3,:3])
        return np.append(t_diff, q_diff, axis=0)

#============================================
# Select Images   
#============================================  

img1, img2 = None, None       # var initialization
img1_box = None               # image 1 bounding box (initialization)
model_fitting_type = None     # 'homography' or 'fundamental' (automatically set below, this is an initialization)
draw_horizontal_layout=True   # draw matches with the two images in an horizontal or vertical layout (automatically set below, this is an initialization) 
gt_pose = None

idx = 20
img1 = cv2.imread(osp.join(uw_root_dir, "images/raw", str(idx) + '_left.png')) # queryImage
img2 = cv2.imread(osp.join(uw_root_dir, "images/raw", str(idx) + '_fish.png')) # trainImage
if img1 is None:
    raise IOError('Cannot find img1')    
if img2 is None: 
    raise IOError('Cannot find img2')  
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

model_fitting_type='essential'
draw_horizontal_layout = True
# Get ground truth pose between stereo and fisheye
gt_pose = read_pose(idx)

#============================================
# Compute keypoints and descriptors  
#============================================  
    
kpts1 = np.array([x.pt for x in kps1], dtype=np.float32) 
kpts2 = np.array([x.pt for x in kps2], dtype=np.float32)

#============================================
# Model fitting for extrapolating inliers 
#============================================  

# Init inliers mask
mask = None 
   
kps1_matched_normalized = normalizeFisheye(kpts1)
kps2_matched_normalized = normalizePerspective(kpts2)
# note threshold is normalized
E, mask = cv2.findEssentialMat(kps1_matched_normalized, kps2_matched_normalized, focal=1, pp=(0., 0.), method=cv2.RANSAC, prob=0.999, threshold=3.0/1800.)
worldpoints, R_est, t_est, mask_pose = cv2.recoverPose(E, kps1_matched_normalized, kps2_matched_normalized)
n_inlier = np.count_nonzero(mask)
q_est = mat2quat(R_est) # wxyz
# refine estimate
# import KnFUSLAM.ransac.pyRansac as erefine
# pose = erefine.newEPose()
# pose.x, pose.y, pose.z = [t_est[i,0] for i in range(len(t_est))]
# pose.qw, pose.qx, pose.qy, pose.qz = [q_est[i] for i in range(len(q_est))]
# refiner = erefine.CRefineEOnRTManifold()
# # Think points get passed backwards 
# pose_ref = refiner.refineRobustOnMask(kps1_matched_normalized, kps2_matched_normalized, mask.flatten(), pose)
print("gt: {}".format(gt_pose[:3]))
print("ransac: {}".format(np.array(t_est*np.linalg.norm(gt_pose[:3])).flatten()))
# print("refined: {}".format(np.array([pose.x, pose.y, pose.z]).flatten()*np.linalg.norm(gt_pose[:3])))
    
    
#============================================
# Drawing  
#============================================  

img_matched_inliers = None 
if mask is not None:    
    # Build arrays of matched inliers 
    mask_idxs = (mask.ravel() == 1)    
    
    kps1_matched_inliers = kps1_matched[mask_idxs]
    kps1_size_inliers = kps1_size[mask_idxs]
    des1_matched_inliers  = des1_matched[mask_idxs][:]    
    kps2_matched_inliers = kps2_matched[mask_idxs]   
    kps2_size_inliers = kps2_size[mask_idxs]    
    des2_matched_inliers  = des2_matched[mask_idxs][:]        
    print('num inliers: ', len(kps1_matched_inliers))
    print('inliers percentage: ', len(kps1_matched_inliers)/max(len(kps1_matched),1.)*100,'%')
        
    sigma_mad_inliers, dists = descriptor_sigma_mad(des1_matched_inliers,des2_matched_inliers,descriptor_distances=feature_tracker.descriptor_distances)
    print('3 x sigma-MAD of descriptor distances (inliers): ', 3 * sigma_mad)  
    #print('distances: ', dists)  
    img_matched_inliers = draw_feature_matches(img1, img2, kps1_matched_inliers, kps2_matched_inliers, kps1_size_inliers, kps2_size_inliers,draw_horizontal_layout)    
                          
                          
img_matched = draw_feature_matches(img1, img2, kps1_matched, kps2_matched, kps1_size, kps2_size,draw_horizontal_layout)
                          
                                                
fig1 = MPlotFigure(img_matched, title='All matches')
if img_matched_inliers is not None: 
    fig2 = MPlotFigure(img_matched_inliers, title='Inlier matches')
MPlotFigure.show()