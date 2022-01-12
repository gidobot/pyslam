from tabulate import tabulate
import numpy as np
from math import acos, sqrt, isnan
from os import path as osp
import csv
from transforms3d.quaternions import mat2quat, qconjugate, qmult, quat2mat
import matplotlib.pyplot as plt
from sklearn.metrics import auc

TRACKERS = ['SIFT', 'ROOT_SIFT', 'ORB', 'SURF', 'AKAZE', 'CONTEXTDESC', 'SUPERPOINT']
# TRACKERS = ['SIFT', 'ROOT_SIFT', 'ORB', 'SUPERPOINT']
# TRACKERS = ['SIFT', 'ROOT_SIFT', 'ORB', 'ORB2', 'AKAZE', 'SUPERPOINT', 'CONTEXTDESC']
# TRACKERS = ['ROOT_SIFT', 'ROOT_SIFT_CLIPPED']
# TRACKERS = ['ROOT_SIFT', 'CONTEXTDESC']

root_dir = "/media/Data/PSTAR/Reconstruction_Datasets/UWHandles/20181221/SlopeAndMagneticSeep/set1/arrangement3/tagslam_old"
# root_dir = "/media/Data/PSTAR/Reconstruction_Datasets/UWHandles/20181221/SlopeAndMagneticSeep/set2/tagslam_old"
# root_dir = "/media/Data/PSTAR/Reconstruction_Datasets/UWHandles/20181220/Mound12/set1/arrangement3/tagslam_old"
# root_dir = "/media/Data/PSTAR/Reconstruction_Datasets/UWHandles/20181219/Mounds1011/set1/arrangement1/tagslam_old"
gt_file = None
input_files = []
with open(osp.join(root_dir, "camera_poses.txt")) as file:
    gt_file = np.array(list(csv.reader(file)), np.double)
for t in TRACKERS:
    with open(osp.join(root_dir, "tracker_poses_"+t+".txt")) as file:
        input_files.append(np.array(list(csv.reader(file)), np.double))
seq_length = input_files[0].shape[0]
# import pdb; pdb.set_trace()
gt_idxs = input_files[0][:,0]

# good plotting colors
colors = ['b', 'g', 'r', 'c', 'm', 'k', 'lime', 'orange', 'peru']

#============================================
# Helper Functions  
#============================================  

def read_gt_pose(idx):
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

def read_poses(idx):
	pose_list = []
	gt_pose = read_gt_pose(int(gt_idxs[idx]))
	pose_list.append(gt_pose)
	for i in range(len(TRACKERS)):
		T = np.append(input_files[i][idx][3:10], [input_files[i][idx][2]], axis=0)
		pose_list.append(T)
	return pose_list

#============================================
# Main 
#============================================ 

t_diffs = []
q_diffs = []
t_means = []
q_means = []
t_med = []
q_med = []
f_count = []
f_means = []
n_matched = []
t_auc = []
q_auc = []
for i in range(len(TRACKERS)):
	t_diffs.append([])
	q_diffs.append([])
	t_means.append([])
	q_means.append([])
	t_med.append([])
	q_med.append([])
	f_count.append([])
	f_means.append([])
	n_matched.append([])
	t_auc.append([])
	q_auc.append([])

for idx in range(seq_length):
	pose_list = read_poses(idx)
	gt_t = pose_list[0][:3]
	gt_t = gt_t/np.linalg.norm(gt_t)
	gt_q = pose_list[0][3:7]
	for i in range(len(TRACKERS)):
		t = pose_list[i+1][:3]
		t = t/np.linalg.norm(t)
		q = pose_list[i+1][3:7]
		td = np.degrees(acos(np.dot(t,gt_t)))
		qd = np.degrees(2*acos(abs(np.dot(q,gt_q))))
		if ~np.isnan(td) and ~np.isnan(qd):
			t_diffs[i].append(td)
			q_diffs[i].append(qd)
			f_count[i].append(pose_list[i+1][7])

# t_diffs = np.array(t_diffs)
# q_diffs = np.array(q_diffs)
# s = np.isnan(t_diffs)
# t_diffs[s] = 180.
# s = np.isnan(q_diffs)
# q_diffs[s] = 180.

# t_diffs = np.sort(t_diffs, axis=1)
# q_diffs = np.sort(q_diffs, axis=1)

sum_array = np.ones(seq_length)
sum_array = np.cumsum(sum_array) / float(seq_length)

for i in range(len(TRACKERS)):
	t_diffs[i] = np.sort(t_diffs[i])
	q_diffs[i] = np.sort(q_diffs[i])

	n_matched[i] = len(t_diffs[i])

	if len(t_diffs[i]) < seq_length:
		t_diffs[i] = np.append(t_diffs[i], [180 for i in range(seq_length - len(t_diffs[i]))], axis=0)
	if len(q_diffs[i]) < seq_length:
		q_diffs[i] = np.append(q_diffs[i], [180 for i in range(seq_length - len(q_diffs[i]))], axis=0)

	t_means[i] = np.mean(t_diffs[i])
	q_means[i] = np.mean(q_diffs[i])
	t_med[i] = np.median(t_diffs[i])
	q_med[i] = np.median(q_diffs[i])

	f_means[i] = np.mean(f_count[i])

	t_auc[i] = 0
	for k in range(1,seq_length):
		t_auc[i] += sum_array[k-1]*(t_diffs[i][k]-t_diffs[i][k-1])
	t_auc[i] += 90. - t_diffs[i][-1]
	t_auc[i] = t_auc[i]/90.

	q_auc[i] = 0
	for k in range(1,seq_length):
		q_auc[i] += sum_array[k-1]*(q_diffs[i][k]-q_diffs[i][k-1])
	q_auc[i] += 180. - q_diffs[i][-1]
	q_auc[i] = q_auc[i]/180.

# t_means = np.mean(t_diffs, axis=1)
# q_means = np.mean(q_diffs, axis=1)
# t_med = np.median(t_diffs, axis=1)
# q_med = np.median(q_diffs, axis=1)

table = [[TRACKERS[i], t_means[i], t_med[i], t_auc[i], q_means[i], q_med[i], q_auc[i], f_means[i], n_matched[i]] for i in range(len(TRACKERS))]
headers = ['Tracker', 't_mean', 't_med', 't_auc', 'q_mean', 'q_med', 'q_auc', 'feat_mean', '# images']
print(tabulate(table, headers=headers))

fig, (ax1, ax2) = plt.subplots(1,2)
fig.set_size_inches(8, 4)
for i in range(len(TRACKERS)):
	ax1.plot(t_diffs[i], sum_array[:len(t_diffs[i])], c=colors[i])
	ax2.plot(q_diffs[i], sum_array[:len(q_diffs[i])], c=colors[i])
ax1.set_title('Translation Direction')
ax1.set_xlabel('Angle Error (deg)')
ax1.set_ylabel('Percentage')
ax1.legend(TRACKERS)
ax2.set_title('Orientation')
ax2.set_xlabel('Angle Error (deg)')
ax2.set_ylabel('Percentage')
# ax2.legend(TRACKERS)

plt.show()