import sys 
import os.path as osp
import numpy as np
import cv2 
from matplotlib import pyplot as plt
from transforms3d.quaternions import mat2quat, qconjugate, qmult, quat2mat
import csv

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

root_dir = "/media/Data/PSTAR/Reconstruction_Datasets/UWHandles/20181221/SlopeAndMagneticSeep/set1/arrangement3/tagslam"
with open(osp.join(root_dir, "camera_poses.txt")) as file:
    input_file = np.array(list(csv.reader(file)), np.double)
output_file = open(osp.join(root_dir, "tracker_poses.txt"),'w')
seq_length = input_file.shape[0]

#============================================
# Helper Functions  
#============================================  

def read_pose(idx):
    # gt tf Translation: [-0.089, -0.364, 1.328]
    #       Rotation: in Quaternion [-0.305, 0.061, 0.149, 0.939]
    row = input_file[idx]
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

# feature tracking config
num_features=2000 

tracker_type = FeatureTrackerTypes.DES_BF      # descriptor-based, brute force matching with knn 
# tracker_type = FeatureTrackerTypes.DES_FLANN  # descriptor-based, FLANN-based matching 

# select your tracker configuration (see the file feature_tracker_configs.py) 
# tracker_config = FeatureTrackerConfigs.TEST
# tracker_config = FeatureTrackerConfigs.SUPERPOINT
tracker_config = FeatureTrackerConfigs.ROOT_SIFT
# tracker_config = FeatureTrackerConfigs.CONTEXTDESC
tracker_config['num_features'] = num_features
tracker_config['match_ratio_test'] = 0.8        # 0.7 is the default in feature_tracker_configs.py
tracker_config['tracker_type'] = tracker_type
print('feature_manager_config: ',tracker_config)

feature_tracker = feature_tracker_factory(**tracker_config)

for idx in range(seq_length):
    print("Processing {}".format(idx))

    img1 = cv2.imread(osp.join(root_dir, "images/raw", str(idx) + '_fish.png')) # queryImage
    img2 = cv2.imread(osp.join(root_dir, "images/raw", str(idx) + '_left.png')) # trainImage
    if img1 is None:
        raise IOError('Cannot find img1')    
        continue
    if img2 is None: 
        raise IOError('Cannot find img2')  
        continue
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    model_fitting_type='essential'
    draw_horizontal_layout = True
    # Get ground truth pose between stereo and fisheye
    gt_pose = read_pose(23)

    # Find the keypoints and descriptors in img1
    kps1, des1 = feature_tracker.detectAndCompute(img1)
    
    # Find the keypoints and descriptors in img2    
    kps2, des2 = feature_tracker.detectAndCompute(img2)
    # Find matches    
    idx1, idx2 = feature_tracker.matcher.match(des1, des2)

    print('#kps1: ', len(kps1))
    if des1 is not None: 
        print('des1 shape: ', des1.shape)
    print('#kps2: ', len(kps2))
    if des2 is not None: 
        print('des2 shape: ', des2.shape)    

    print('number of matches: ', len(idx1))

    # Convert from list of keypoints to an array of points 
    kpts1 = np.array([x.pt for x in kps1], dtype=np.float32) 
    kpts2 = np.array([x.pt for x in kps2], dtype=np.float32)

    # Get keypoint size 
    kps1_size = np.array([x.size for x in kps1], dtype=np.float32)  
    kps2_size = np.array([x.size for x in kps2], dtype=np.float32) 

    # Build arrays of matched keypoints, descriptors, sizes 
    kps1_matched = kpts1[idx1]
    des1_matched = des1[idx1][:]
    kps1_size = kps1_size[idx1]

    kps2_matched = kpts2[idx2]
    des2_matched = des2[idx2][:]
    kps2_size = kps2_size[idx2]

    # compute sigma mad of descriptor distances
    # sigma_mad, dists = descriptor_sigma_mad(des1_matched,des2_matched,descriptor_distances=feature_tracker.descriptor_distances)
    # print('3 x sigma-MAD of descriptor distances (all): ', 3 * sigma_mad)

    # Init inliers mask
    mask = None 
    pose_est = None
       
    h1,w1 = img1.shape[:2]  
    if kps1_matched.shape[0] > 10:
        kps1_matched_normalized = normalizeFisheye(kps1_matched)
        kps2_matched_normalized = normalizePerspective(kps2_matched)
        # note threshold is normalized
        E, mask = cv2.findEssentialMat(kps1_matched_normalized, kps2_matched_normalized, focal=1, pp=(0., 0.), method=cv2.RANSAC, prob=0.999, threshold=3.0/729.)
        worldpoints, R_est, t_est, mask_pose = cv2.recoverPose(E, kps1_matched_normalized, kps2_matched_normalized)
        n_inlier = np.count_nonzero(mask)
        q_est = mat2quat(R_est).flatten() # wxyz
        t_est = t_est.flatten()*np.linalg.norm(gt_pose[:3])
        pose_est = np.append(t_est, q_est, axis=0)
    else:
        mask = None 
        print('Not enough matches are found for', model_fitting_type)
        continue 

    mask_idxs = (mask.ravel() == 1)    
    kps1_matched_inliers = kps1_matched[mask_idxs]
    print('num inliers: ', len(kps1_matched_inliers))
    print('inliers percentage: ', len(kps1_matched_inliers)/max(len(kps1_matched),1.)*100,'%')

    # write pose file
    output_file.write("{},{},{},{},{},{},{},{}\n".format(
        idx, pose_est[0], pose_est[1], pose_est[2], pose_est[3], pose_est[4], pose_est[5], pose_est[6]))
    output_file.flush()

# close files
output_file.close()

def draw_imgs():
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