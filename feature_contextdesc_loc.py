"""
* This file is part of PYSLAM 
*
* Adpated from https://github.com/lzx551402/contextdesc/blob/master/image_matching.py, see the license therein. 
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com> 
*
* PYSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* PYSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.
"""

import config
config.cfg.set_lib('contextdesc',prepend=True) 

from threading import RLock

import warnings # to disable tensorflow-numpy warnings: from https://github.com/tensorflow/tensorflow/issues/30427
warnings.filterwarnings('ignore', category=FutureWarning)

import os
import cv2
import numpy as np

# if True:
#     import tensorflow as tf
# else: 
#     # from https://stackoverflow.com/questions/56820327/the-name-tf-session-is-deprecated-please-use-tf-compat-v1-session-instead
#     import tensorflow.compat.v1 as tf

from contextdesc.utils.opencvhelper import MatcherWrapper

#from contextdesc.models import get_model
from contextdesc.models.loc_model import LocModel 

# from utils_tf import set_tf_logging
#from utils_sys import Printer

kVerbose = True   
    
# convert matrix of pts into list of keypoints
def convert_pts_to_keypoints(pts, scores, sizes): 
    assert(len(pts)==len(scores))
    kps = []
    if pts is not None: 
        # convert matrix [Nx2] of pts into list of keypoints  
        kps = [ cv2.KeyPoint(p[0], p[1], _size=sizes[i], _response=scores[i]) for i,p in enumerate(pts) ]                      
    return kps         

# interface for pySLAM 
class ContextDescFeature2D: 
    quantize=False      #  Whether to quantize or not the output descriptor 
    def __init__(self,
                 num_features=2000,
                 n_sample=2048,              #  Maximum number of sampled keypoints per octave
                 model_type='trt',                  
                 do_tf_logging=False):  
        print('Using ContextDescFeature2D')   
        self.lock = RLock()
        self.model_base_path= config.cfg.root_folder + '/thirdparty/contextdesc/'
        
        # set_tf_logging(do_tf_logging)
        
        self.num_features = num_features
        self.n_sample = n_sample
        self.model_type = model_type
        self.quantize = ContextDescFeature2D.quantize
        
        self.loc_model_path = self.model_base_path + 'pretrained/contextdesc++'

        self.grid_batch = False
            
        if self.model_type == 'pb':
            loc_model_path = os.path.join(self.loc_model_path, 'loc.pb')
        elif self.model_type == 'pbv2':
            loc_model_path = os.path.join(self.loc_model_path, 'retrained/model.pb')
        elif self.model_type == 'ckpt':
            loc_model_path = os.path.join(self.loc_model_path, 'model.ckpt-400000')
        elif self.model_type == 'tflite':
            # loc_model_path = os.path.join(self.loc_model_path, 'loc_quant.tflite')
            loc_model_path = os.path.join(self.loc_model_path, 'loc_quant_keras.tflite')
            # loc_model_path = os.path.join(self.loc_model_path, 'loc_quant_keras_grid.tflite')
        elif self.model_type == 'tpu':
            # loc_model_path = os.path.join(self.loc_model_path, 'loc_quant_keras_edgetpu.tflite')
            loc_model_path = os.path.join(self.loc_model_path, 'loc_quant_keras_grid_edgetpu.tflite')
        elif self.model_type == 'trt':
            loc_model_path = os.path.join(self.loc_model_path, 'descnet_trt.engine')
        elif self.model_type == 'keras':
            # loc_model_path = os.path.join(self.loc_model_path, 'descnet.hdf5')
            # loc_model_path = os.path.join(self.loc_model_path, 'descnet_lite.hdf5')
            # loc_model_path = os.path.join(self.loc_model_path, 'descnet_liter.hdf5')
            loc_model_path = os.path.join(self.loc_model_path, 'descnet_litest.hdf5')
            # loc_model_path = os.path.join(self.loc_model_path, 'descnet_litest_grid.hdf5')
            # loc_model_path = os.path.join(self.loc_model_path, 'descnet_litest2.hdf5')
            # loc_model_path = os.path.join(self.loc_model_path, 'descnet_litext.hdf5')
            # loc_model_path = os.path.join(self.loc_model_path, 'descnet_litext2.hdf5')
            print("Using model: {}".format(self.loc_model_path))
        else:
            raise NotImplementedError
        
        self.keypoint_size = 10  # just a representative size for visualization and in order to convert extracted points to cv2.KeyPoint        

        self.pts = []
        self.kps = []        
        self.des = []
        self.scales = []
        self.scores = []        
        self.frame = None 
        
        print('==> Loading pre-trained network.')
        self.loc_model = LocModel(loc_model_path, **{'sift_desc': False,             # compute or not SIFT descriptor (we do not need them here!)
                                                    'n_feature': self.num_features,                                                     
                                                    'n_sample': self.n_sample,
                                                    'peak_thld': 0.04,
                                                    'dense_desc': False,
                                                    'model_type': self.model_type,
                                                    'grid_batch': self.grid_batch,
                                                    'upright': False})       
        print('==> Successfully loaded pre-trained network.')
            
    def __del__(self): 
        with self.lock:              
            self.loc_model.close()
                
    def prep_img(self,img):
        rgb_list = []
        gray_list = []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[..., np.newaxis]
        img = img[..., ::-1]
        rgb_list.append(img)
        gray_list.append(gray)
        return rgb_list, gray_list                


    # extract local features and keypoint matchability
    def extract_local_features(self,gray_list):
        cv_kpts_list = []
        loc_info_list = []
        loc_feat_list = []
        sift_feat_list = []
        # model = get_model('loc_model')(model_path, **{'sift_desc': True,
        #                                             'n_sample': FLAGS.n_sample,
        #                                             'peak_thld': 0.04,
        #                                             'dense_desc': FLAGS.dense_desc,
        #                                             'upright': False})        
        for _, val in enumerate(gray_list):
            loc_feat, kpt_mb, normalized_xy, cv_kpts, sift_desc = self.loc_model.run_test_data(val)
            raw_kpts = [np.array((i.pt[0], i.pt[1], i.size, i.angle, i.response)) for i in cv_kpts]
            raw_kpts = np.stack(raw_kpts, axis=0)
            loc_info = np.concatenate((raw_kpts, normalized_xy, loc_feat, kpt_mb), axis=-1)
            cv_kpts_list.append(cv_kpts)
            loc_info_list.append(loc_info)
            sift_feat_list.append(sift_desc)
            loc_feat_list.append(loc_feat / np.linalg.norm(loc_feat, axis=-1, keepdims=True))
        #model.close()
        return cv_kpts_list, loc_info_list, loc_feat_list, sift_feat_list

    def compute_kps_des(self, frame):
        with self.lock:         
            rgb_list, gray_list = self.prep_img(frame)
            cv_kpts_list, loc_info_list, loc_feat_list, sift_feat_list = self.extract_local_features(gray_list)

            self.kps = cv_kpts_list[0]
            self.des = loc_feat_list[0]
            
            return self.kps, self.des   
        
           
    def detectAndCompute(self, frame, mask=None):  #mask is a fake input  
        with self.lock: 
            self.frame = frame         
            self.kps, self.des = self.compute_kps_des(frame)            
            if kVerbose:
                print('detector: CONTEXTDESC, descriptor: CONTEXTDESC, #features: ', len(self.kps), ', frame res: ', frame.shape[0:2])                  
            return self.kps, self.des
    
           
    # return keypoints if available otherwise call detectAndCompute()    
    def detect(self, frame, mask=None):  # mask is a fake input
        with self.lock:           
            if self.frame is not frame:
                self.detectAndCompute(frame)        
            return self.kps
    
    
    # return descriptors if available otherwise call detectAndCompute()  
    def compute(self, frame, kps=None, mask=None): # kps is a fake input, mask is a fake input
        with self.lock:         
            if self.frame is not frame:
                #Printer.orange('WARNING: CONTEXTDESC is recomputing both kps and des on last input frame', frame.shape)            
                self.detectAndCompute(frame)
            return self.kps, self.des   
