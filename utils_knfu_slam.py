import numpy as np
import cv2
from collections import Counter, OrderedDict
from matplotlib import pyplot as plt

def clipScaleRatioSIFT(idx1, idx2, kps1, kps2):
    idx1 = np.array(idx1)
    idx2 = np.array(idx2)
    scales1 = np.array([kps1[idx1[i]].octave for i in range(len(idx1))], dtype=np.float32) 
    scales2 = np.array([kps2[idx2[i]].octave for i in range(len(idx2))], dtype=np.float32) 
    sr = np.array([scales2[i] - scales1[i] for i in range(scales1.size)])
    hist = np.array(list(OrderedDict(sorted(Counter(sr).items())).items()), np.int)
    m_sr_idx = np.argmax(hist[:,1])
    m_sr = hist[m_sr_idx,0]
    clip_idx = np.where((sr >= m_sr-1) & (sr <= m_sr+1))
    idx1 = idx1[clip_idx]
    idx2 = idx2[clip_idx]
    return idx1, idx2

def scaleImprovedSIFT(img1, img2, feature_tracker):
    # Initial matching to find image scale difference
    kps1, des1 = feature_tracker.detectAndCompute(img1)
    kps2, des2 = feature_tracker.detectAndCompute(img2)
    idx1, idx2 = feature_tracker.matcher.match(des1, des2)
    # Convert from list of keypoints to an array of scales 
    scales1 = np.array([x.octave for x in kps1], dtype=np.float32) 
    scales2 = np.array([x.octave for x in kps2], dtype=np.float32)
    scales1_matched = scales1[idx1]
    scales2_matched = scales2[idx2]
    sr = [scales2_matched[i] - scales1_matched[i] for i in range(scales1_matched.size)]
    hist = np.array(list(OrderedDict(sorted(Counter(sr).items())).items()), np.int)
    print(hist)
    plt.plot(hist[:,0],hist[:,1])
    plt.show()


# N.B.1: Assume distortion is insignificant for now for feature point matching. ASLAM provides undistort functions for equidistant model.
def normalizeFisheye(kps, camera_matrix=None, dist_coeffs=None):
    camera_matrix = np.array([[769.5519232429078, 0.000000, 1268.7948261550591], [0.000000, 768.8322015619301, 1023.8486413295748], [0.000000, 0.000000, 1.000000]])
    dist_coeffs = np.array([0.025513637146558403, -0.011386600859137145, 0.013688146542497151, -0.0076052132438100654])
    kps = np.ascontiguousarray(kps, np.float32)
    kps_norm = np.squeeze(cv2.fisheye.undistortPoints(np.expand_dims(kps, axis=1), camera_matrix, dist_coeffs))
    # f = focal_length
    # c = image_center
    # k = dist_coeffs
    # kps_c = kps - c
    # # add small value to r for numerical stability
    # r = np.sqrt(kps_c[:,0]**2 + kps_c[:,1]**2) + 1e-6
    # theta = r/f
    # z = r/np.tan(theta)
    # kps_norm = kps_c/np.tile(np.expand_dims(z,axis=1),(1,2))
    return kps_norm

def normalizePerspective(kps, camera_matrix=None, dist_coeffs=None):
    camera_matrix = np.array([[1836.115648, 0.000000, 1213.199687], [0.000000, 1836.768364, 988.405167], [0.000000, 0.000000, 1.000000]])
    dist_coeffs = np.array([-0.005789, -0.021328, -0.001998, -0.002663])
    kps = np.ascontiguousarray(kps, np.float32)
    kps_norm = np.squeeze(cv2.undistortPoints(np.expand_dims(kps, axis=1), cameraMatrix=camera_matrix, distCoeffs=dist_coeffs))
    return kps_norm

# def estimate_relative_pose_from_correspondence(kps, camera_matrix, dist_coeffs):
#         f_avg = (K1[0, 0] + K2[0, 0]) / 2
#         pts1, pts2 = np.ascontiguousarray(pts1, np.float32), np.ascontiguousarray(pts2, np.float32)

#         pts_l_norm = cv2.undistortPoints(np.expand_dims(pts1, axis=1), cameraMatrix=K1, distCoeffs=None)
#         pts_r_norm = cv2.undistortPoints(np.expand_dims(pts2, axis=1), cameraMatrix=K2, distCoeffs=None)

#         E, mask = cv2.findEssentialMat(pts_l_norm, pts_r_norm, focal=1.0, pp=(0., 0.),
#                                        method=cv2.RANSAC, prob=0.999, threshold=3.0 / f_avg)
#         points, R_est, t_est, mask_pose = cv2.recoverPose(E, pts_l_norm, pts_r_norm)
#         return mask[:,0].astype(np.bool), R_est, t_est 

# fit essential matrix E with RANSAC such that:  p2.T * E * p1 = 0  where  E = [t21]x * R21
# input: kpn_ref and kpn_cur are two arrays of [Nx2] normalized coordinates of matched keypoints 
# out: a) Trc: homogeneous transformation matrix containing Rrc, trc  ('cur' frame with respect to 'ref' frame)    pr = Trc * pc 
#      b) mask_match: array of N elements, every element of which is set to 0 for outliers and to 1 for the other points (computed only in the RANSAC and LMedS methods)
# N.B.1: trc is estimated up to scale (i.e. the algorithm always returns ||trc||=1, we need a scale in order to recover a translation which is coherent with previous estimated poses)
# N.B.2: this function has problems in the following cases: [see Hartley/Zisserman Book]
# - 'geometrical degenerate correspondences', e.g. all the observed features lie on a plane (the correct model for the correspondences is an homography) or lie a ruled quadric 
# - degenerate motions such a pure rotation (a sufficient parallax is required) or an infinitesimal viewpoint change (where the translation is almost zero)
# N.B.3: the five-point algorithm (used for estimating the Essential Matrix) seems to work well in the degenerate planar cases [Five-Point Motion Estimation Made Easy, Hartley]
# N.B.4: as reported above, in case of pure rotation, this algorithm will compute a useless fundamental matrix which cannot be decomposed to return a correct rotation 
# N.B.5: the OpenCV findEssentialMat function uses the five-point algorithm solver by D. Nister => hence it should work well in the degenerate planar cases
# def findEssentialMat(kp1, kp2, method=cv2.RANSAC, prob=0.999, threshold=0.0003):  
#     # here, the essential matrix algorithm uses the five-point algorithm solver by D. Nister (see the notes and paper above )     
#     E, mask_match = cv2.findEssentialMat(kp1, kp2, focal=1, pp=(0., 0.), method=method, prob=prob, threshold=threshold)                         
#     _, R, t, mask = cv2.recoverPose(E, kp1, kp2, focal=1, pp=(0., 0.))   
#     return poseRt(R,t.T), mask_match  # Trc, mask_mat         

# def find_essential_mat(kp1, kp2, method=cv2.RANSAC, prob=0.999, threshold=0.0003):  
#     # here, the essential matrix algorithm uses the five-point algorithm solver by D. Nister (see the notes and paper above )     
#     E, mask_match = cv2.findEssentialMat(kp1, kp2, focal=1, pp=(0., 0.), method=method, prob=prob, threshold=threshold)                         
#     _, R, t, mask = cv2.recoverPose(E, kp1, kp2, focal=1, pp=(0., 0.))   
#     return poseRt(R,t.T), mask_match  # Trc, mask_mat         

# estimate a pose from a fitted essential mat; 
# since we do not have an interframe translation scale, this fitting can be used to detect outliers, estimate interframe orientation and translation direction 
# N.B. read the NBs of the method estimate_pose_ess_mat(), where the limitations of this method are explained  
# def estimate_pose_by_fitting_ess_mat(self, f_ref, f_cur, idxs_ref, idxs_cur): 
#     # N.B.: in order to understand the limitations of fitting an essential mat, read the comments of the method self.estimate_pose_ess_mat() 
#     self.timer_pose_est.start()
#     # estimate inter frame camera motion by using found keypoint matches 
#     # output of the following function is:  Trc = [Rrc, trc] with ||trc||=1  where c=cur, r=ref  and  pr = Trc * pc 
#     Mrc, self.mask_match = estimate_pose_ess_mat(f_ref.kpsn[idxs_ref], f_cur.kpsn[idxs_cur], 
#                                                  method=cv2.RANSAC, prob=kRansacProb, threshold=kRansacThresholdNormalized)   
#     #Mcr = np.linalg.inv(poseRt(Mrc[:3, :3], Mrc[:3, 3]))   
#     Mcr = inv_T(Mrc)
#     estimated_Tcw = np.dot(Mcr, f_ref.pose)
#     self.timer_pose_est.refresh()      

#     # remove outliers from keypoint matches by using the mask computed with inter frame pose estimation        
#     mask_idxs = (self.mask_match.ravel() == 1)
#     self.num_inliers = sum(mask_idxs)
#     print('# inliers: ', self.num_inliers )
#     idxs_ref = idxs_ref[mask_idxs]
#     idxs_cur = idxs_cur[mask_idxs]

#     # if there are not enough inliers do not use the estimated pose 
#     if self.num_inliers < kNumMinInliersEssentialMat:
#         #f_cur.update_pose(f_ref.pose) # reset estimated pose to previous frame 
#         Printer.red('Essential mat: not enough inliers!')  
#     else:
#         # use the estimated pose as an initial guess for the subsequent pose optimization 
#         # set only the estimated rotation (essential mat computation does not provide a scale for the translation, see above) 
#         #f_cur.pose[:3,:3] = estimated_Tcw[:3,:3] # copy only the rotation 
#         #f_cur.pose[:,3] = f_ref.pose[:,3].copy() # override translation with ref frame translation 
#         Rcw = estimated_Tcw[:3,:3] # copy only the rotation 
#         tcw = f_ref.pose[:3,3]     # override translation with ref frame translation          
#         f_cur.update_rotation_and_translation(Rcw, tcw)     
#     return  idxs_ref, idxs_cur