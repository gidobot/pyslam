"""
Copyright 2020, Zixin Luo, HKUST.
Image matching example.
"""

import config
config.cfg.set_lib('ASLFeat',prepend=True) 

from threading import RLock

import warnings # to disable tensorflow-numpy warnings: from https://github.com/tensorflow/tensorflow/issues/30427
warnings.filterwarnings('ignore', category=FutureWarning)

import os
import cv2
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = ""

if False:
    import tensorflow as tf
else: 
    # from https://stackoverflow.com/questions/56820327/the-name-tf-session-is-deprecated-please-use-tf-compat-v1-session-instead
    import tensorflow.compat.v1 as tf

from ASLFeat.utils.opencvhelper import MatcherWrapper
from ASLFeat.models.feat_model import FeatModel
# from ASLFeat.models import get_model

from utils_tf import set_tf_logging

kVerbose = True

# interface for pySLAM 
class ASLFeature2D: 
    def __init__(self,
                 num_features=2000,
                 model_type='ckpt',                  
                 do_tf_logging=False):  
        print('Using ASLFeat')   
        self.lock = RLock()
        self.model_base_path= config.cfg.root_folder + '/thirdparty/ASLFeat/'
        
        set_tf_logging(do_tf_logging)
        
        self.num_features = num_features
        self.model_type = model_type
        
        self.model_path = self.model_base_path + 'pretrained/aslfeatv2'
            
        # if self.model_type == 'pb':
            # self.model_path = os.path.join(self.model_path, 'aslfeat.pb')
        if self.model_type == 'ckpt':
            self.model_path = os.path.join(self.model_path, 'model.ckpt-60000')
        else:
            print("Model not found at path {}".format(self.model_path))
            raise NotImplementedError
        
        self.keypoint_size = 10  # just a representative size for visualization and in order to convert extracted points to cv2.KeyPoint        

        self.kps = []        
        self.des = []
        self.frame = None 
        
        print('==> Loading pre-trained network.')
        config_loc = {'max_dim': 2048,
                      'config':{
                      'kpt_n': self.num_features,
                      'kpt_refinement': True,
                      'deform_desc': 1,
                      'score_thld': 0.5,
                      'edge_thld': 10,
                      'multi_scale': True,
                      'multi_level': True,
                      'nms_size': 3,
                      'eof_mask': 5,
                      'need_norm': True,
                      'use_peakiness': True}}
        self.model = FeatModel(self.model_path, **config_loc)
        # self.model = get_model('feat_model')(self.model_path, **config_loc)
        print('==> Successfully loaded pre-trained network.')
            

    def __del__(self): 
        with self.lock:              
            self.model.close()                
                
                
    def prep_img(self,img):
        rgb_list = []
        gray_list = []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[..., np.newaxis]
        img = img[..., ::-1]
        rgb_list.append(img)
        gray_list.append(gray)
        return rgb_list, gray_list                

        
    def compute_kps_des(self, frame):
        with self.lock:         
            rgb_list, gray_list = self.prep_img(frame)
            # extract features.
            des, kps, _ = self.model.run_test_data(gray_list[0])
        return kps, des
        
           
    def detectAndCompute(self, frame, mask=None):  #mask is a fake input  
        with self.lock: 
            self.frame = frame         
            self.kps, self.des = self.compute_kps_des(frame)            
            if kVerbose:
                print('detector: ASLFeat, descriptor: ASLFeat, #features: ', len(self.kps), ', frame res: ', frame.shape[0:2])                  
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
                #Printer.orange('WARNING: ASLFeat is recomputing both kps and des on last input frame', frame.shape)            
                self.detectAndCompute(frame)
            return self.kps, self.des   