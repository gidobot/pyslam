import sys 
import os.path as osp
import numpy as np
import cv2 
from matplotlib import pyplot as plt
from collections import Counter, OrderedDict
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
from utils_knfu_slam import normalizeFisheye, normalizePerspective, clipScaleRatioSIFT

from feature_tracker_configs import FeatureTrackerConfigs


# TRACKERS = ['SIFT', 'ROOT_SIFT', 'ORB', 'SURF', 'AKAZE', 'CONTEXTDESC']
TRACKERS = ['SUPERPOINT']
# TRACKERS = ['ORB']
# TRACKERS = ['AKAZE']
# TRACKERS = ['CONTEXTDESC']
# TRACKERS = ['ROOT_SIFT', 'SIFT']
# TRACKERS = ['SURF_SIFT']
# TRACKERS = ['SURF']

# root_dir = "/media/Data/PSTAR/Reconstruction_Datasets/UWHandles/20181221/SlopeAndMagneticSeep/set1/arrangement3/tagslam_old"
root_dir = "/media/Data/PSTAR/Reconstruction_Datasets/UWHandles/20181221/SlopeAndMagneticSeep/set2/tagslam_old"
# root_dir = "/media/Data/PSTAR/Reconstruction_Datasets/UWHandles/20181220/Mound12/set1/arrangement3/tagslam_old"
# root_dir = "/media/Data/PSTAR/Reconstruction_Datasets/UWHandles/20181219/Mounds1011/set1/arrangement1/tagslam_old"
gt_file = None
with open(osp.join(root_dir, "camera_poses.txt")) as file:
    gt_file = np.array(list(csv.reader(file)), np.double)
seq_length = gt_file.shape[0]
# output_files = [open(osp.join(root_dir, "tracker_poses_"+t+"_CLIPPED"+".txt"),'w') for t in TRACKERS]
# output_files = [open(osp.join(root_dir, "tracker_poses_"+t+"_opt.txt"),'w') for t in TRACKERS]
output_files = [open(osp.join(root_dir, "tracker_poses_"+t+".txt"),'w') for t in TRACKERS]

#============================================
# Helper Functions  
#============================================  

def draw_imgs():
    img_matched_inliers = None 
    draw_horizontal_layout = True
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
            
        sigma_mad_inliers, dists = descriptor_sigma_mad(des1_matched_inliers,des2_matched_inliers,descriptor_distances=feature_trackers[i].descriptor_distances)
        print('3 x sigma-MAD of descriptor distances (inliers): ', 3 * sigma_mad_inliers)  
        #print('distances: ', dists)  
        img_matched_inliers = draw_feature_matches(img1, img2, kps1_matched_inliers, kps2_matched_inliers, kps1_size_inliers, kps2_size_inliers,draw_horizontal_layout)    
    img_matched = draw_feature_matches(img1, img2, kps1_matched, kps2_matched, kps1_size, kps2_size, draw_horizontal_layout)
    # fig1 = MPlotFigure(img_matched, title='All matches')
    if img_matched_inliers is not None: 
        fig2 = MPlotFigure(img_matched_inliers, title='Inlier matches')
    MPlotFigure.show()

def read_pose(idx):
    row = gt_file[idx]
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
    # take tf difference between stereo and fisheye
    T_diff = T_left.dot(T_fish)
    t_diff = T_diff[:3,3]
    q_diff = mat2quat(T_diff[:3,:3])
    return np.append(t_diff, q_diff, axis=0)

# feature tracking config
num_features=2000 

tracker_type = FeatureTrackerTypes.DES_BF      # descriptor-based, brute force matching with knn 
# tracker_type = FeatureTrackerTypes.DES_FLANN  # descriptor-based, FLANN-based matching 

tracker_configs = [getattr(FeatureTrackerConfigs, t) for t in TRACKERS]
for i, t in enumerate(tracker_configs):
    t['num_features'] = num_features
    t['match_ratio_test'] = 0.8        # 0.7 is the default in feature_tracker_configs.py
    t['tracker_type'] = tracker_type
    if TRACKERS[i] == 'CONTEXTDESC':
        t['match_ratio_test'] = 0.9        # 0.7 is the default in feature_tracker_configs.py
    print('feature_manager_config: ',t)

feature_trackers = [feature_tracker_factory(**t) for t in tracker_configs]

scale_ratios = []
step = 5
for idx in range(0, seq_length, step):
    print("Processing {}".format(idx))

    # img1 = cv2.imread(osp.join(root_dir, "images/raw", str(idx) + '_fish.png')) # queryImage
    # img2 = cv2.imread(osp.join(root_dir, "images/raw", str(idx) + '_left.png')) # trainImage
    img1 = cv2.imread(osp.join(root_dir, "images/masked", str(idx) + '_fish.png')) # queryImage
    img2 = cv2.imread(osp.join(root_dir, "images/masked", str(idx) + '_left.png')) # trainImage
    if img1 is None:
        raise IOError('Cannot find img1')    
        continue
    if img2 is None: 
        raise IOError('Cannot find img2')  
        continue
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # Find the keypoints and descriptors in img1
    for i in range(len(TRACKERS)):
        print("\n{}: {} of {}".format(TRACKERS[i],int(idx/step),int(seq_length/step)))
        # Find the keypoints and descriptors in img2    
        kps1, des1 = feature_trackers[i].detectAndCompute(img1)
        kps2, des2 = feature_trackers[i].detectAndCompute(img2)
        # Find matches    
        idx1, idx2 = feature_trackers[i].matcher.match(des1, des2)

        print('#kps1: ', len(kps1))
        if des1 is not None: 
            print('des1 shape: ', des1.shape)
        print('#kps2: ', len(kps2))
        if des2 is not None: 
            print('des2 shape: ', des2.shape)    
        print('number of matches: ', len(idx1))

        # ROOT_SIFT scale ratios
        # if TRACKERS[i] == 'ROOT_SIFT':
        #     # idx1, idx2 = clipScaleRatioSIFT(idx1, idx2, kps1, kps2)
        #     scales1 = np.array([x.octave for x in kps1], dtype=np.float32) 
        #     scales2 = np.array([x.octave for x in kps2], dtype=np.float32)
        #     scales1_matched = scales1[idx1]
        #     scales2_matched = scales2[idx2]
        #     sr = [scales2_matched[i] - scales1_matched[i] for i in range(scales1_matched.size)]
        #     scale_ratios = scale_ratios + sr

        # Convert from list of keypoints to an array of points 
        kps1 = np.array([x.pt for x in kps1], dtype=np.float32) 
        kps2 = np.array([x.pt for x in kps2], dtype=np.float32)

        # Get keypoint size 
        kps1_size = np.array([x.size for x in kps1], dtype=np.float32)  
        kps2_size = np.array([x.size for x in kps2], dtype=np.float32) 

        # Build arrays of matched keypoints, descriptors, sizes 
        kps1_matched = kps1[idx1]
        des1_matched = des1[idx1][:]
        kps1_size = kps1_size[idx1]

        kps2_matched = kps2[idx2]
        des2_matched = des2[idx2][:]
        kps2_size = kps2_size[idx2]

        # Init inliers mask
        mask = None 
        pose_est = pose_est = np.zeros(7)
           
        h1,w1 = img1.shape[:2]  
        kps1_matched_inliers = []
        if kps1_matched.shape[0] > 10:
            kps1_matched_normalized = normalizeFisheye(kps1_matched)
            kps2_matched_normalized = normalizePerspective(kps2_matched)
            # note threshold is normalized
            E, mask = cv2.findEssentialMat(kps1_matched_normalized, kps2_matched_normalized, focal=1, pp=(0., 0.), method=cv2.RANSAC, prob=0.999, threshold=3.0/729.)
            worldpoints, R_est, t_est, mask_pose = cv2.recoverPose(E, kps1_matched_normalized, kps2_matched_normalized)
            n_inlier = np.count_nonzero(mask)
            q_est = mat2quat(R_est).flatten() # wxyz
            t_est = t_est.flatten()
            pose_est = np.append(t_est/np.linalg.norm(t_est), q_est, axis=0)

            mask_idxs = (mask.ravel() == 1)    
            kps1_matched_inliers = kps1_matched[mask_idxs]
            print('num inliers: ', len(kps1_matched_inliers))
            print('inliers percentage: ', len(kps1_matched_inliers)/max(len(kps1_matched),1.)*100,'%')
            # draw_imgs()
        else:
            mask = None 
            print('Not enough matches are found')

        # write pose file
        pose_list = [str(e) for e in [idx, len(kps1_matched), len(kps1_matched_inliers),
            pose_est[0], pose_est[1], pose_est[2], pose_est[3], pose_est[4], pose_est[5], pose_est[6]]]
        output_files[i].write(",".join(pose_list)+'\n')
        output_files[i].flush()

# hist = np.array(list(OrderedDict(sorted(Counter(scale_ratios).items())).items()), np.int)
# plt.plot(hist[:,0],hist[:,1])
# plt.show()

# close files
for file in output_files:
    file.close()
