#First load libraries and images
# %matplotlib inline
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import kornia
import cv2
from kornia.feature import *
from time import time
import torch.optim as optim
from torch.nn import Parameter
from kornia.color import rgb_to_grayscale

img1 = Image.open('/home/gidobot/workspace/KnFU-SLAM/pyslam/test/data/underwater/23_fish.png')
img2 = Image.open('/home/gidobot/workspace/KnFU-SLAM/pyslam/test/data/underwater/23_left.png')#.resize(img1.size)

timg1 = kornia.utils.image_to_tensor(np.array(img1), False).float() / 255.
timg2 = kornia.utils.image_to_tensor(np.array(img2), False).float() / 255.

timg = torch.cat([timg1,timg2], dim=0)

timg_gray = kornia.color.rgb_to_grayscale(timg)

plt.imshow(kornia.utils.tensor_to_image(timg[0]))
plt.figure()
plt.imshow(kornia.utils.tensor_to_image(timg[1]))

#Lets define some functions for local feature matching

def visualize_LAF(img, LAF, img_idx = 0):
    x, y = kornia.feature.laf.get_laf_pts_to_draw(LAF, img_idx)
    plt.figure()
    plt.imshow(kornia.utils.tensor_to_image(img[img_idx]))
    plt.plot(x, y, 'r')
    plt.show()
    return

# N.B.1: Assume distortion is insignificant for now for feature point matching. ASLAM provides undistort functions for equidistant model.
def normalizeFisheye(kps, camera_matrix=None, dist_coeffs=None):
    camera_matrix = np.array([[769.5519232429078, 0.000000, 1268.7948261550591], [0.000000, 768.8322015619301, 1023.8486413295748], [0.000000, 0.000000, 1.000000]])
    dist_coeffs = np.array([0.025513637146558403, -0.011386600859137145, 0.013688146542497151, -0.0076052132438100654])
    kps = np.ascontiguousarray(kps, np.float32)
    kps_norm = np.squeeze(cv2.fisheye.undistortPoints(np.expand_dims(kps, axis=1), camera_matrix, dist_coeffs))
    return kps_norm

def normalizePerspective(kps, camera_matrix=None, dist_coeffs=None):
    camera_matrix = np.array([[1836.115648, 0.000000, 1213.199687], [0.000000, 1836.768364, 988.405167], [0.000000, 0.000000, 1.000000]])
    dist_coeffs = np.array([-0.005789, -0.021328, -0.001998, -0.002663])
    kps = np.ascontiguousarray(kps, np.float32)
    kps_norm = np.squeeze(cv2.undistortPoints(np.expand_dims(kps, axis=1), cameraMatrix=camera_matrix, distCoeffs=dist_coeffs))
    return 

#Now lets define local deature detector and descriptor


device = torch.device('cpu')
# device = torch.device('cuda:0')

PS = 41

sift = kornia.feature.SIFTDescriptor(PS, rootsift=True).to(device)
descriptor = sift

resp = BlobHessian()
scale_pyr = kornia.geometry.ScalePyramid(3, 1.6, PS, double_image=True)

nms = kornia.geometry.ConvQuadInterp3d(10)

n_features = 4000
detector = ScaleSpaceDetector(n_features,
                              resp_module=resp,
                              nms_module=nms,
                              scale_pyr_module=scale_pyr,
                              ori_module=kornia.feature.LAFOrienter(19),
                              mr_size=6.0).to(device)

with torch.no_grad():
    lafs, resps = detector(timg_gray)
    patches =  kornia.feature.extract_patches_from_pyramid(timg_gray, lafs, PS)
    B, N, CH, H, W = patches.size()
    # Descriptor accepts standard tensor [B, CH, H, W], while patches are [B, N, CH, H, W] shape
    # So we need to reshape a bit :) 
    descs = descriptor(patches.view(B * N, CH, H, W)).view(B, N, -1)
    scores, matches = kornia.feature.match_snn(descs[0], descs[1], 0.9)

# Now RANSAC
src_pts = lafs[0,matches[:,0], :, 2].data.cpu().numpy()
dst_pts = lafs[1,matches[:,1], :, 2].data.cpu().numpy()
src_pts_n = normalizeFisheye(src_pts)
dst_pots_n = normalizePerspective(dst_pts)


# H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 1.0, 0.999, 10000)
E, mask = cv2.findEssentialMat(src_pts_n, dst_pts_n, focal=1, pp=(0., 0.), method=cv2.RANSAC, prob=0.999, threshold=1./729.)

inliers = matches[torch.from_numpy(mask).bool().squeeze(), :]

for i in range(2):
    visualize_LAF(timg_gray, lafs[:,inliers[:,i]], i)
print (len(inliers), 'inliers')


# #Now lets define  affine local deature detector and descriptor

# device = torch.device('cpu')
# # device = torch.device('cuda:0')

# PS = 41

# sift = kornia.feature.SIFTDescriptor(PS, rootsift=True).to(device)
# descriptor = sift

# resp = BlobHessian()
# scale_pyr = kornia.geometry.ScalePyramid(3, 1.6, PS, double_image=True)

# nms = kornia.geometry.ConvQuadInterp3d(10)

# n_features = 4000
# detector = ScaleSpaceDetector(n_features,
#                               resp_module=resp,
#                               nms_module=nms,
#                               scale_pyr_module=scale_pyr,
#                               ori_module=kornia.feature.LAFOrienter(19),
#                               aff_module=kornia.feature.LAFAffineShapeEstimator(19),
#                               mr_size=6.0).to(device)

# with torch.no_grad():
#     lafs, resps = detector(timg_gray)
#     patches =  kornia.feature.extract_patches_from_pyramid(timg_gray, lafs, PS)
#     B, N, CH, H, W = patches.size()
#     # Descriptor accepts standard tensor [B, CH, H, W], while patches are [B, N, CH, H, W] shape
#     # So we need to reshape a bit :) 
#     descs = descriptor(patches.view(B * N, CH, H, W)).view(B, N, -1)
#     scores, matches = kornia.feature.match_snn(descs[0], descs[1], 0.9)

# # Now RANSAC
# src_pts = lafs[0,matches[:,0], :, 2].data.cpu().numpy()
# dst_pts = lafs[1,matches[:,1], :, 2].data.cpu().numpy()

# # H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 1.0, 0.999, 10000)

# # inliers = matches[torch.from_numpy(mask).bool().squeeze(), :]

# # for i in range(2):
# #     visualize_LAF(timg_gray, lafs[:,inliers[:,i]], i)
# # print (len(inliers), 'inliers')

# #Now lets try with Difference-of-Gaussians, aka SIFT keypoints

# device = torch.device('cpu')
# #device = torch.device('cuda:0')

# PS = 41

# sift = kornia.feature.SIFTDescriptor(PS, rootsift=True).to(device)
# descriptor = sift

# resp = BlobDoG()
# scale_pyr = kornia.geometry.ScalePyramid(3, 1.6, PS, double_image=True)

# nms = kornia.geometry.ConvQuadInterp3d(10)

# n_features = 4000
# detector = ScaleSpaceDetector(n_features,
#                               resp_module=resp,
#                               scale_space_response=True,#We need that, because DoG operates on scale-space
#                               nms_module=nms,
#                               scale_pyr_module=scale_pyr,
#                               ori_module=kornia.feature.LAFOrienter(19),
#                               mr_size=6.0,
#                               minima_are_also_good=True).to(device) #dark blobs are as good as bright.

# with torch.no_grad():
#     lafs, resps = detector(timg_gray)
#     patches =  kornia.feature.extract_patches_from_pyramid(timg_gray, lafs, PS)
#     B, N, CH, H, W = patches.size()
#     # Descriptor accepts standard tensor [B, CH, H, W], while patches are [B, N, CH, H, W] shape
#     # So we need to reshape a bit :) 
#     descs = descriptor(patches.view(B * N, CH, H, W)).view(B, N, -1)
#     scores, matches = kornia.feature.match_snn(descs[0], descs[1], 0.9)

# # Now RANSAC
# src_pts = lafs[0,matches[:,0], :, 2].data.cpu().numpy()
# dst_pts = lafs[1,matches[:,1], :, 2].data.cpu().numpy()

# # H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 1.0, 0.999, 10000)

# # inliers = matches[torch.from_numpy(mask).bool().squeeze(), :]

# # for i in range(2):
# #     visualize_LAF(timg_gray, lafs[:,inliers[:,i]], i)
# # print (len(inliers), 'inliers')