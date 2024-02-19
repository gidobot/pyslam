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
import csv

if False:
    import tensorflow as tf
else: 
    # from https://stackoverflow.com/questions/56820327/the-name-tf-session-is-deprecated-please-use-tf-compat-v1-session-instead
    import tensorflow.compat.v1 as tf


from contextdesc.utils.opencvhelper import MatcherWrapper

#from contextdesc.models import get_model
from contextdesc.models.loc_model import LocModel 

from utils_tf import set_tf_logging
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
    quantize=False      #  Wheter to quantize or not the output descriptor 
    def __init__(self,
                 num_features=2000,
                 n_sample=2048,              #  Maximum number of sampled keypoints per octave
                 model_type='pb',                  
                 do_tf_logging=False):  
        print('Using ContextDescFeature2D')   
        self.lock = RLock()
        self.model_base_path= config.cfg.root_folder + '/thirdparty/contextdesc/'

        self.data_dir = '/media/gidobot/data/UWslam_dataset/hybrid/Mounds1/'
        
        set_tf_logging(do_tf_logging)
        
        self.num_features = num_features
        self.n_sample = n_sample
        self.model_type = model_type
        self.quantize = ContextDescFeature2D.quantize
        
        self.loc_model_path = self.model_base_path + 'pretrained/contextdesc++/'
            
        if self.model_type == 'pb':
            # loc_model_path = os.path.join(self.loc_model_path, 'loc.pb')
            loc_model_path = os.path.join(self.loc_model_path, 'retrained/model.pb')
        elif self.model_type == 'ckpt':
            loc_model_path = os.path.join(self.loc_model_path, 'model.ckpt-400000')
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

    def representative_data_gen(self):
        gt_file = None
        with open(os.path.join(self.data_dir, "camera_poses.txt")) as file:
            gt_file = np.array(list(csv.reader(file)), np.double)
        seq_length = gt_file.shape[0]
        step = 50
        for idx in range(0, seq_length//step, step):
            image = cv2.imread(os.path.join(self.data_dir, "images/raw", str(idx) + '_left.png')) # queryImage
            rgb_list, gray_list = self.prep_img(image)
            gray_img = np.squeeze(gray_list[0], axis=-1).astype(np.uint8)
            # detect SIFT keypoints.
            self.loc_model.sift_wrapper.build_pyramid(gray_img)
            npy_kpts, cv_kpts = self.loc_model.sift_wrapper.detect(gray_img)
            all_patches = self.loc_model.sift_wrapper.get_patches(cv_kpts)
            all_patches = np.expand_dims(all_patches, -1)
            print("Yielding patch set {} with {} patches".format(idx, len(all_patches)))
            yield [all_patches]

    def quantize(self):
        # converter = tf.lite.TFLiteConverter.from_frozen_graph(self.loc_model_path+'/loc.pb',
            # input_arrays = ['input'], output_arrays = ['conv6_feat'], input_shapes={'input': [2000,32,32,1]})
        converter = tf.lite.TFLiteConverter.from_frozen_graph(self.loc_model_path+'/retrained/model.pb',
            input_arrays = ['input/net_input'], output_arrays = ['feat_tower0/conv6/Conv2D'], input_shapes={'input/net_input': [2000,32,32,1]})
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # This ensures that if any ops can't be quantized, the converter throws an error
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # These set the input and output tensors to uint8
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        # And this sets the representative dataset so we can quantize the activations
        converter.representative_dataset = self.representative_data_gen
        tflite_model = converter.convert()

        with open(self.loc_model_path+'/loc_quant.tflite', 'wb') as f:
          f.write(tflite_model)

if __name__ == '__main__':
    model = ContextDescFeature2D()
    model.quantize(model)