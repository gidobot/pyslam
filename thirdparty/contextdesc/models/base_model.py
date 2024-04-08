#!/usr/bin/env python3

"""
Copyright 2019, Zixin Luo, HKUST.
Inference script.
"""

import sys
import os
from abc import ABCMeta, abstractmethod
import collections
import numpy as np
# import tensorflow as tf
# import tflite_runtime.interpreter as tflite
#import pycuda.driver as cuda
#import pycuda.autoinit

sys.path.append('..')

def dict_update(d, u):
    """Improved update for nested dictionaries.

    Arguments:
        d: The dictionary to be updated.
        u: The update dictionary.

    Returns:
        The updated dictionary.
    """
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


class BaseModel(metaclass=ABCMeta):
    """Base model class."""

    @abstractmethod
    def _run(self, data, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _init_model(self):
        raise NotImplementedError

    @abstractmethod
    def _construct_network(self):
        raise NotImplementedError

    def run_test_data(self, data, **kwargs):
        """"""
        out_data = self._run(data, **kwargs)
        return out_data

    

    def __init__(self, model_path, **config):
        self.model_path = model_path
        self.interpreter = None
        self.model = None
        # Update config
        self.config = dict_update(getattr(self, 'default_config', {}), config)
        self._init_model()
        if model_path is None:
            print("No pretrained model specified!")
            self.sess = None
        elif '.engine' in model_path:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
            TRT_LOGGER = trt.Logger()

            def load_engine(engine_file_path):
                assert os.path.exists(engine_file_path)
                print("Reading engine from file {}".format(engine_file_path))
                with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                    return runtime.deserialize_cuda_engine(f.read()) 
            self.sess = None
            self.interpreter = None
            self.config['model_type'] = 'trt'
            self.engine = load_engine(model_path)
            self.context = self.engine.create_execution_context()
            # Set input shape based on feature patches for inference
            self.context.set_binding_shape(0, (self.config['n_feature'], 32, 32, 1))
            # Allocate host and device buffers
            self.bindings = []
            dummy_input = np.zeros((2000,32,32,1), dtype=np.float16)
            for binding in self.engine:
               binding_idx = self.engine.get_binding_index(binding)
               size = trt.volume(self.context.get_binding_shape(binding_idx))
               dtype = trt.nptype(self.engine.get_binding_dtype(binding))
               if self.engine.binding_is_input(binding):
                   self.input_memory = cuda.mem_alloc(dummy_input.nbytes)
                   self.bindings.append(int(self.input_memory))
               else:
                   self.output_buffer = cuda.pagelocked_empty(size, dtype)
                   self.output_memory = cuda.mem_alloc(self.output_buffer.nbytes)
                   self.bindings.append(int(self.output_memory))
            self.stream = cuda.Stream()
        elif 'edgetpu.tflite' in model_path:
            from pycoral.utils.edgetpu import make_interpreter

            self.sess = None
            self.config['model_type'] = 'tpu'
            # self.interpreter = tflite.Interpreter(self.model_path, experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
            # self.interpreter = tf.lite.Interpreter(self.model_path, experimental_delegates=[tf.lite.experimental.load_delegate('libedgetpu.so.1')])
            self.interpreter = make_interpreter(self.model_path)
            # self.interpreter.resize_tensor_input(0, [self.config['n_feature'], 32, 32, 1])
            self.interpreter.allocate_tensors()
        else:
            import tensorflow as tf
            from ..utils.tf import load_frozen_model, recoverer

            ext = os.path.splitext(model_path)[1]

            sess_config = tf.compat.v1.ConfigProto()
            sess_config.gpu_options.allow_growth = True
            sess_config.gpu_options.per_process_gpu_memory_fraction=0.333

            if ext.find('.pb') == 0:
                graph = load_frozen_model(self.model_path, print_nodes=False)
                self.sess = tf.compat.v1.Session(graph=graph, config=sess_config)
            elif ext.find('.ckpt') == 0:
                self.sess = tf.compat.v1.Session(config=sess_config)
                meta_graph_path = os.path.join(model_path + '.meta')
                if not os.path.exists(meta_graph_path):
                    self._construct_network()
                    recoverer(self.sess, model_path)
                else:
                    recoverer(self.sess, model_path, meta_graph_path)
            elif ext.find('.tflite') == 0:
                self.sess = None
                # self.config['tflite'] = True
                self.config['model_type'] = 'tflite'
                # self.interpreter = tf.compat.v1.lite.Interpreter(self.model_path)
                self.interpreter = tf.lite.Interpreter(self.model_path)
                # self.interpreter.resize_tensor_input(0, [self.config['n_feature'], 32, 32, 1])
                self.interpreter.allocate_tensors()
            elif ext.find('.hdf5') == 0:
                self.sess = None
                self.model = tf.keras.models.load_model(self.model_path)
            else:
                print("Unknown model type: {}".format(ext))
                raise Exception("Unknown model type: {}".format(ext))


    def close(self):
        if self.sess is not None:
            self.sess.close()
            tf.compat.v1.reset_default_graph()
