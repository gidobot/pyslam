#!/usr/bin/env python3
"""
Copyright 2019, Zixin Luo, HKUST.
Image matching example.
"""
import os
import cv2
import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from utils.opencvhelper import MatcherWrapper

from models import get_model


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('reg_model', 'pretrained/retrieval_model',
                           """Path to the regional feature model.""")
tf.app.flags.DEFINE_string('data_path', 'data',
                           """Path to the first image.""")
# model options
tf.app.flags.DEFINE_string('model_type', 'pb',
                           """Pre-trained model type.""")


def load_img(img_path):
    img = cv2.imread(img_path)
    img = img[..., ::-1]
    max_dim = max(img.shape[0], img.shape[1])
    downsample_ratio = 448 / float(max_dim)
    img = cv2.resize(img, (0, 0), fx=downsample_ratio, fy=downsample_ratio)
    return img

def main(argv=None):  # pylint: disable=unused-argument
    """Program entrance."""
    if FLAGS.model_type == 'pb':
        reg_model_path = os.path.join(FLAGS.reg_model, 'reg.pb')
    elif FLAGS.model_type == 'ckpt':
        reg_model_path = os.path.join(FLAGS.reg_model, 'model.ckpt-550000')
    else:
        raise NotImplementedError

    reg_model = get_model('reg_model')(reg_model_path)

    reg_feat_list = []
    data_dirs = glob.glob(FLAGS.data_path + "/*")
    for n, d in enumerate(data_dirs):
        img_list = glob.glob(os.path.join(d, 'undist_images/*.jpg'))
        if not os.path.exists(os.path.join(d, 'reg_feat')):
            os.makedirs(os.path.join(d, 'reg_feat'))
        for i, img_path in enumerate(img_list):
            name = os.path.splitext(os.path.basename(img_path))[0]
            save_path = os.path.join(d, 'reg_feat', name+'.bin')
            print('Processing image {} of {}, in folder {} of {}: {}'.format(i, len(img_list), n, len(data_dirs), save_path))
            if os.path.exists(save_path):
                continue
            rgb = load_img(img_path)
            reg_feat = reg_model.run_test_data(rgb)
            reg_feat.astype(np.float32).tofile(save_path)

    reg_model.close()

if __name__ == '__main__':
    tf.compat.v1.app.run()
