import numpy as np
import tensorflow as tf

from .base_model import BaseModel
from .cnn_wrapper.augdesc import VisualContext, LightContextNormalization


class AugModel(BaseModel):
    output_tensors = "l2norm:0"
    default_config = {'quantz': False, 'use_sift': True, 'reg_feat_dim': 2048}

    def _init_model(self):
        return

    def _run(self, data):
        reg_feat = data[0]
        loc_info = data[1]
        raw_kpts = loc_info[:, 0:5]
        npy_kpts = loc_info[:, 5:7]
        if not self.config['use_sift']:
            loc_feat = loc_info[:, 7:-1]
            kpt_mb = loc_info[:, -1][..., np.newaxis]
        else:
            loc_feat = loc_info[:, 7:]
            # self.output_tensors = ["l2norm:0", "aug_tower0/kpt_m/kpt_m/BiasAdd:0"]

        ### Test
        # # Get the default graph
        # graph = tf.compat.v1.get_default_graph()

        # # List tensors that have input
        # input_tensors = []
        # for op in graph.get_operations():
        #     if op.type == 'Placeholder':
        #         input_tensors.append(op.name)

        # print("Input Tensors:")
        # for tensor_name in input_tensors:
        #     print(tensor_name)

        # uninitialized_vars = []
        # for var in tf.compat.v1.global_variables():
        #     if self.sess.run(tf.compat.v1.is_variable_initialized(var)) == False:
        #         uninitialized_vars.append(var)
        # print("Uninitialized Variables:")
        # for var in uninitialized_vars:
        #     print(var.name)

        # import pdb; pdb.set_trace()
        ### Test

        # self.sess.run(tf.compat.v1.global_variables_initializer())
        # all_vars = tf.compat.v1.global_variables()
        # for var in all_vars:
            # if 'kpt_' in var.name:
                # print(var.name)

        if not self.config['use_sift']:
            returns = self.sess.run(self.output_tensors, feed_dict={
                "input/local_feat:0": np.expand_dims(loc_feat, 0),
                "input/regional_feat:0": np.expand_dims(reg_feat, 0),
                "input/kpt_m:0": np.expand_dims(kpt_mb, 0),
                "input/kpt_xy:0": np.expand_dims(npy_kpts, 0),
            })
            aug_feat = np.squeeze(returns, axis=0)
        else:
            returns = self.sess.run(self.output_tensors, feed_dict={
                "input/local_feat:0": np.expand_dims(loc_feat, 0),
                "input/regional_feat:0": np.expand_dims(reg_feat, 0),
                "input/kpt_xy:0": np.expand_dims(npy_kpts, 0),
            })
            # aug_feat = returns[0]
            # kpt_mb = returns[1]
            aug_feat = np.squeeze(returns, axis=0)

        if self.config['quantz']:
            aug_feat = (aug_feat * 128 + 128).astype(np.uint8)
        return aug_feat, raw_kpts

    def _construct_network(self):
        with tf.name_scope('input'):
            ph_local_feat = tf.placeholder(
                dtype=tf.float32, shape=(None, None, 128), name='local_feat')
            ph_regional_feat = tf.placeholder(dtype=tf.float32, shape=(
                None, None, None, self.config['reg_feat_dim']), name='regional_feat')
            ph_kpt_m = tf.placeholder(dtype=tf.float32, shape=(None, None, 1), name='kpt_m')
            ph_kpt_xy = tf.placeholder(dtype=tf.float32, shape=(
                None, None, 2), name='kpt_xy')

        aug_feat = [ph_local_feat]

        bs = tf.shape(ph_regional_feat)[0]
        rows = tf.shape(ph_regional_feat)[1]
        cols = tf.shape(ph_regional_feat)[2]
        x_rng = tf.linspace(-1., 1., cols)
        y_rng = tf.linspace(-1., 1., rows)
        xv, yv = tf.meshgrid(x_rng, y_rng)
        grid_pts = tf.stack((xv, yv), axis=-1)
        grid_pts = tf.expand_dims(grid_pts, axis=0)
        grid_pts = tf.tile(grid_pts, (bs, 1, 1, 1))

        with tf.variable_scope('vis_context'):
            pt_tower = VisualContext(
                {'grid_pts': grid_pts, 'img_feat': ph_regional_feat,
                 'local_feat': ph_local_feat, 'kpt_param': ph_kpt_xy},
                is_training=False, reuse=False)
            vis_feat = pt_tower.get_output()
            aug_feat.append(vis_feat)

        with tf.variable_scope('geo_context'):
            points = tf.concat([ph_kpt_m, ph_kpt_xy], axis=2)
            feat_trans = LightContextNormalization({'points': tf.expand_dims(points, axis=2),
                                                    'local_feat': ph_local_feat},
                                                   is_training=False, reuse=False)
            geo_feat = feat_trans.get_output()
            aug_feat.append(geo_feat)

        aug_feat = tf.add_n(aug_feat)
        aug_feat = tf.nn.l2_normalize(aug_feat, axis=-1, name='l2norm')
