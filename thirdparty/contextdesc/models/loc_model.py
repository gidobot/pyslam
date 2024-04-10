import sys
from queue import Queue
from threading import Thread
import time

import os
from struct import unpack
import numpy as np
import cv2

# import pycuda.driver as cuda
# import pycuda.autoinit
# import tensorrt as trt

from .base_model import BaseModel

sys.path.append('..')

from ..utils.opencvhelper import SiftWrapper


class LocModel(BaseModel):
    output_tensors = ["conv6_feat:0", "kpt_mb:0"]
    default_config = {'n_feature': 0, "n_sample": 0,
                      'batch_size': 2000, 'sift_wrapper': None, 'upright': False, 'scale_diff': False,
                      'dense_desc': False, 'sift_desc': False, 'peak_thld': 0.0067, 'edge_thld': 10, 'max_dim': 1280}

    def __init__(self, *args, **kwargs):
        super(LocModel,self).__init__(*args, **kwargs)
        if self.config['model_type'] == 'trt':
            self.cuda_driver_context.push()
        #     # Allocate host and device buffers
        #     self.bindings = []
        #     dummy_input = np.zeros((2000,32,32,1), dtype=np.float16)
        #     for binding in self.engine:
        #         binding_idx = self.engine.get_binding_index(binding)
        #         size = trt.volume(self.context.get_binding_shape(binding_idx))
        #         dtype = trt.nptype(self.engine.get_binding_dtype(binding))
        #         if self.engine.binding_is_input(binding):
        #             self.input_memory = cuda.mem_alloc(dummy_input.nbytes)
        #             self.bindings.append(int(self.input_memory))
        #         else:
        #             self.output_buffer = cuda.pagelocked_empty(size, dtype)
        #             self.output_memory = cuda.mem_alloc(self.output_buffer.nbytes)
        #             self.bindings.append(int(self.output_memory))
        #     self.stream = cuda.Stream()

    def __del__(self):
        if self.config['model_type'] == 'trt':
            self.cuda_driver_context.pop()

    def _init_model(self):
        self.sift_wrapper = SiftWrapper(
            n_feature=self.config['n_feature'],
            n_sample=self.config['n_sample'],
            peak_thld=self.config['peak_thld'],
            edge_thld=self.config['edge_thld']
            )
        self.sift_wrapper.standardize = False  # the network has handled this step.
        self.sift_wrapper.ori_off = self.config['upright']
        self.sift_wrapper.pyr_off = not self.config['scale_diff']
        self.sift_wrapper.create()
        
    def _set_input_tensor_tflite(self, input):
        input_details = self.interpreter.get_input_details()[0]
        tensor_index = input_details['index']
        input_tensor = self.interpreter.tensor(tensor_index)()
        # Inputs for the TFLite model must be int8, so we quantize our input data.
        # NOTE: This step is necessary only because we're receiving input data from
        # ImageDataGenerator, which rescaled all image data to float [0,1]. When using
        # bitmap inputs, they're already uint8 [0,255] so this can be replaced with:
        # input_tensor[:, :, :, :] = input
        # import pdb; pdb.set_trace()
        scale, zero_point = input_details['quantization']
        input_tensor[:, :, :, :] = np.int8(input / scale + zero_point)
        # input_tensor[:, :, :, :] = np.uint8(input / scale + zero_point)

    def _run_tflite(self, input):
        batch_size = input.shape[0]
        self._set_input_tensor_tflite(input)
        start = time.perf_counter()
        self.interpreter.invoke()
        end = time.perf_counter()
        print("Time to compute {} tflite descriptors: {}".format(batch_size, end - start))
        output_details = self.interpreter.get_output_details()[0]
        output = self.interpreter.get_tensor(output_details['index'])
        output = np.squeeze(output)
        # Outputs from the TFLite model are int8, so we dequantize the results:
        scale, zero_point = output_details['quantization']
        output = scale * (output.astype(np.float32) - zero_point)
        # Only return outputs for the input batch size
        # output = np.asarray(output[:batch_size], dtype=np.float32)
        # output = np.asarray(output, dtype=np.float32)
        return output

    def _run_tpu(self, input):
        from pycoral.adapters import common
        # batch_size = input.shape[0]
        input_details = self.interpreter.get_input_details()[0]
        output_details = self.interpreter.get_output_details()[0]
        scale, zero_point = input_details['quantization']
        input_quant = np.int8(input / scale + zero_point)
        # output_quant = np.empty((batch_size, 128), dtype=np.int8)
        common.set_input(self.interpreter, input_quant)
        start = time.perf_counter()
        self.interpreter.invoke()
        end = time.perf_counter()
        print("Time to compute 2000 tpu descriptors: {}ms".format((end - start)*1000))
        output_quant = common.output_tensor(self.interpreter, 0)
        output_quant = np.squeeze(output_quant)
        # for i in range(batch_size):
        #     common.set_input(self.interpreter, input_quant[i])
        #     start = time.perf_counter()
        #     self.interpreter.invoke()
        #     end = time.perf_counter()
        #     print("Time to compute 1 tpu descriptor: {}ms".format((end - start)*1000))
        #     output_quant[i] = common.output_tensor(self.interpreter, 0)[0,0,0,:]
        # Outputs from the TFLite model are int8, so we dequantize the results:
        scale, zero_point = output_details['quantization']
        output = scale * (output_quant.astype(np.float32) - zero_point)
        # Only return outputs for the input batch size
        # output = np.asarray(output[:batch_size], dtype=np.float32)
        return output

    def _run_keras(self, input):
        start = time.perf_counter()
        output = self.model(input)
        end = time.perf_counter()
        print("Time to compute 2000 keras descriptors: {}ms".format((end - start)*1000))
        output = np.squeeze(output)
        return output

    def _run_trt(self, input):
        input_buffer = np.ascontiguousarray(input.astype(np.float32))
        # self.cuda_driver_context.push()
        start = time.perf_counter()
        # Transfer input data to the GPU.
        self.cuda.memcpy_htod_async(self.input_memory, input_buffer, self.stream)
        # Run inference
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        # Transfer prediction output from the GPU.
        self.cuda.memcpy_dtoh_async(self.output_buffer, self.output_memory, self.stream)
        # Synchronize the stream
        self.stream.synchronize()
        end = time.perf_counter()
        # self.cuda_driver_context.pop()
        print("Time to compute 2000 TensorRT descriptors: {}ms".format((end - start)*1000))
        # return np.reshape(self.output_buffer.astype(np.float32), (-1, 128))
        return np.reshape(self.output_buffer, (-1, 128))

    def _run(self, data, **kwargs):
        def _worker(patch_queue, sess, loc_feat, kpt_mb):
            """The worker thread."""
            while True:
                patch_data = patch_queue.get()
                if patch_data is None:
                    return
                batch_size = patch_data.shape[0]
                if batch_size < 2000:
                    # append zeros to make the input batch size 2000
                    patch_data = np.concatenate((patch_data, np.zeros((2000 - patch_data.shape[0], patch_data.shape[1], patch_data.shape[2]))), axis=0)
                if self.config['grid_batch']:
                    shape = patch_data.shape
                    patch_grid = np.zeros((2*shape[0] - 1, 32,  32))
                    idx = list(range(0, patch_grid.shape[0], 2))
                    patch_grid[idx] = patch_data
                    patch_data = np.reshape(patch_grid, (1, -1, 32))

                if self.config['model_type'] == 'tflite':
                    loc_returns = self._run_tflite(np.expand_dims(patch_data, -1))
                    loc_returns = loc_returns[:batch_size]
                    loc_feat.append(loc_returns)
                    kpt_mb.append(np.ones((loc_returns.shape[0], 1)))
                elif self.config['model_type'] == 'trt':
                    loc_returns = self._run_trt(np.expand_dims(patch_data, -1))
                    loc_returns = loc_returns[:batch_size]
                    loc_feat.append(loc_returns)
                    kpt_mb.append(np.ones((loc_returns.shape[0], 1)))
                elif self.config['model_type'] == 'tpu':
                    loc_returns = self._run_tpu(np.expand_dims(patch_data, -1))
                    loc_returns = loc_returns[:batch_size]
                    loc_feat.append(loc_returns)
                    kpt_mb.append(np.ones((loc_returns.shape[0], 1)))
                elif self.config['model_type'] == 'keras':
                    loc_returns = self._run_keras(np.expand_dims(patch_data, -1))
                    loc_returns = loc_returns[:batch_size]
                    loc_feat.append(loc_returns)
                    kpt_mb.append(np.ones((loc_returns.shape[0], 1)))
                elif self.config['model_type'] == 'pbv2':
                    self.output_tensors = ["feat_tower0/conv6/Conv2D:0"]
                    loc_returns = sess.run(self.output_tensors,
                                           feed_dict={"input/net_input:0": np.expand_dims(patch_data, -1)})
                    # squeeze dimensions
                    loc_returns = np.squeeze(loc_returns[0])
                    loc_returns = loc_returns[:batch_size]
                    loc_feat.append(loc_returns)
                    kpt_mb.append(np.ones((loc_returns.shape[0], 1)))
                else:
                    loc_returns = sess.run(self.output_tensors,
                                           feed_dict={"input:0": np.expand_dims(patch_data, -1)})
                    loc_returns = loc_returns[:,:batch_size,:]
                    loc_feat.append(loc_returns[0])
                    kpt_mb.append(loc_returns[1])
                patch_queue.task_done()
        gray_img = np.squeeze(data, axis=-1).astype(np.uint8)
        # detect SIFT keypoints.
        npy_kpts, cv_kpts = self.sift_wrapper.detect(gray_img)
        if self.config['sift_desc']:
            sift_desc = self.sift_wrapper.compute(gray_img, cv_kpts)
        else:
            sift_desc = None

        kpt_xy = np.stack(((npy_kpts[:, 0] - gray_img.shape[1] / 2.) / (gray_img.shape[1] / 2.),
                           (npy_kpts[:, 1] - gray_img.shape[0] / 2.) / (gray_img.shape[0] / 2.)),
                          axis=-1)

        num_patch = len(cv_kpts)

        if not self.config['dense_desc']:
            self.sift_wrapper.build_pyramid(gray_img)
            all_patches = self.sift_wrapper.get_patches(cv_kpts)
            # get iteration number
            batch_size = self.config['batch_size']
            if num_patch % batch_size > 0:
                loop_num = int(np.floor(float(num_patch) / float(batch_size)))
            else:
                loop_num = int(num_patch / batch_size - 1)
            # create input thread
            loc_feat = []
            kpt_mb = []
            patch_queue = Queue()
            worker_thread = Thread(target=_worker, args=(patch_queue, self.sess, loc_feat, kpt_mb))
            worker_thread.daemon = True
            worker_thread.start()
            # enqueue
            for i in range(loop_num + 1):
                if i < loop_num:
                    patch_queue.put(all_patches[i * batch_size: (i + 1) * batch_size])
                else:
                    patch_queue.put(all_patches[i * batch_size:])
            # poison pill
            patch_queue.put(None)
            # wait for extraction.
            worker_thread.join()
            loc_feat = np.concatenate(loc_feat, axis=0)
            kpt_mb = np.concatenate(kpt_mb, axis=0)

        else:
            import cv2
            # compose affine crop matrix.
            patch_scale = 6
            patch_param = np.zeros((num_patch, 6))

            m_cos = np.cos(npy_kpts[:, 3]) * patch_scale * npy_kpts[:, 2]
            m_sin = np.sin(npy_kpts[:, 3]) * patch_scale * npy_kpts[:, 2]

            short_side = float(min(gray_img.shape[0], gray_img.shape[1]))

            patch_param[:, 0] = m_cos / short_side
            patch_param[:, 1] = m_sin / short_side
            patch_param[:, 2] = kpt_xy[:, 0]
            patch_param[:, 3] = -m_sin / short_side
            patch_param[:, 4] = m_cos / short_side
            patch_param[:, 5] = kpt_xy[:, 1]

            max_dim = max(gray_img.shape[0], gray_img.shape[1])
            if max_dim > self.config['max_dim']:
                downsample_ratio = self.config['max_dim'] / float(max_dim)
                gray_img = cv2.resize(gray_img, (0, 0), fx=downsample_ratio, fy=downsample_ratio)

            gray_img = gray_img[..., np.newaxis]
            input_dict = {"input/img:0": np.expand_dims(gray_img, 0),
                          "input/kpt_param:0": np.expand_dims(patch_param, 0)}
            local_returns = self.sess.run(self.output_tensors, feed_dict=input_dict)
            loc_feat = local_returns[0]
            kpt_mb = local_returns[1]

        return loc_feat, kpt_mb, kpt_xy, cv_kpts, sift_desc

    def _construct_network(self):
        """Model for patch description."""
        import tensorflow as tf
        from .cnn_wrapper.descnet import GeoDesc, DenseGeoDesc
        from .cnn_wrapper.augdesc import MatchabilityPrediction
        from ..utils.spatial_transformer import transformer_crop


        if self.config['dense_desc']:
            with tf.name_scope('input'):
                ph_imgs = tf.placeholder(dtype=tf.float32, shape=(
                    None, None, None, 1), name='img')
                ph_kpt_params = tf.placeholder(tf.float32, shape=(None, None, 6), name='kpt_param')
            kpt_xy = tf.concat((ph_kpt_params[:, :, 2, None], ph_kpt_params[:, :, 5, None]), axis=-1)
            kpt_theta = tf.reshape(ph_kpt_params, (tf.shape(ph_kpt_params)[0], tf.shape(ph_kpt_params)[1], 2, 3))
            mean, variance = tf.nn.moments(
                tf.cast(ph_imgs, tf.float32), axes=[1, 2], keep_dims=True)
            norm_input = tf.nn.batch_normalization(ph_imgs, mean, variance, None, None, 1e-5)
            config_dict = {}
            config_dict['pert_theta'] = kpt_theta
            config_dict['patch_sampler'] = transformer_crop
            tower = DenseGeoDesc({'data': norm_input, 'kpt_coord': kpt_xy},
                          is_training=False, resue=False, **config_dict)
        else:
            input_size = (32, 32)
            patches = tf.placeholder(
                dtype=tf.float32, shape=(None, input_size[0], input_size[1], 1), name='input')
            # patch standardization
            mean, variance = tf.nn.moments(
                tf.cast(patches, tf.float32), axes=[1, 2], keep_dims=True)
            patches = tf.nn.batch_normalization(patches, mean, variance, None, None, 1e-5)
            tower = GeoDesc({'data': patches}, is_training=False, reuse=False)

        conv6_feat = tower.get_output_by_name('conv6')
        conv6_feat = tf.squeeze(conv6_feat, axis=[1, 2], name='conv6_feat')

        with tf.compat.v1.variable_scope('kpt_m'):
            inter_feat = tower.get_output_by_name('conv5')
            matchability_tower = MatchabilityPrediction(
                {'data': inter_feat}, is_training=False, reuse=False)
            mb = matchability_tower.get_output()
        mb = tf.squeeze(mb, axis=[1, 2], name='kpt_mb')
