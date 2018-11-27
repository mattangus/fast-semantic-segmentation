r"""Run inference on an image or group of images."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import timeit
import numpy as np
from PIL import Image
import tensorflow as tf
from google.protobuf import text_format
import glob
import matplotlib.pyplot as plt
import cv2
import torch
from sklearn.feature_extraction import image
import copy
import tensorflow_probability as tfp
from abc import ABC, abstractmethod

import extractors

from libs import sliding_window
from protos import pipeline_pb2
from builders import model_builder, dataset_builder
from libs.exporter import deploy_segmentation_inference_graph
from libs.constants import CITYSCAPES_LABEL_COLORS, CITYSCAPES_LABEL_IDS
from libs.custom_metric import streaming_mean
from submod.cholesky.cholesky_update import cholesky_update

slim = tf.contrib.slim

prefetch_queue = slim.prefetch_queue

resnet_ex_class = extractors.pspnet_icnet_resnet_v1.PSPNetICNetDilatedResnet50FeatureExtractor
mobilenet_ex_class = extractors.pspnet_icnet_mobilenet_v2.PSPNetICNetMobilenetV2FeatureExtractor

flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_string('input_shape', '1024,2048,3', # default Cityscapes values
                    'The shape to use for inference. This should '
                    'be in the form [height, width, channels]. A batch '
                    'dimension is not supported for this test script.')

flags.DEFINE_string('patch_size', None, '')

flags.DEFINE_string('pad_to_shape', '1025,2049', # default Cityscapes values
                     'Pad the input image to the specified shape. Must have '
                     'the shape specified as [height, width].')

flags.DEFINE_string('config_path', None,
                    'Path to a pipeline_pb2.TrainEvalPipelineConfig config '
                    'file.')

flags.DEFINE_string('trained_checkpoint', None,
                    'Path to trained checkpoint, typically of the form '
                    'path/to/model.ckpt')

flags.DEFINE_string('output_dir', None, 'Path to write outputs images.')

flags.DEFINE_boolean('label_ids', False,
                     'Whether the output should be label ids.')

def create_input(tensor_dict,
                batch_size,
                batch_queue_capacity,
                batch_queue_threads,
                prefetch_queue_capacity):

    def cast_and_reshape(tensor_dict, dicy_key):
        items = tensor_dict[dicy_key]
        float_images = tf.to_float(items)
        tensor_dict[dicy_key] = float_images
        return tensor_dict

    tensor_dict = cast_and_reshape(tensor_dict,
                    dataset_builder._IMAGE_FIELD)

    batched_tensors = tf.train.batch(tensor_dict,
        batch_size=batch_size, num_threads=batch_queue_threads,
        capacity=batch_queue_capacity, dynamic_pad=True,
        allow_smaller_final_batch=False)

    return prefetch_queue.prefetch_queue(batched_tensors,
        capacity=prefetch_queue_capacity,
        dynamic_pad=False)

class StatComputer(ABC):
    
    @abstractmethod
    def get_update_op(self):
        pass
    
    @abstractmethod
    def save_variable(self, sess, stat_dir):
        pass

class MeanComputer(StatComputer):

    def __init__(self, values, weights):
        self.values = values
        self.weights = weights

        with tf.variable_scope("MeanComputer"):
            self.mean, self.update = streaming_mean(self.values, self.weights, True)
            self.mean = tf.expand_dims(self.mean,0)
    
    def get_update_op(self):
        return self.update
    
    def save_variable(self, sess, file_name):
        mean_value = sess.run(self.mean)
        if np.isnan(mean_value).any():
            print("nan time")
            import pdb; pdb.set_trace()
        print("saving to", file_name)
        np.savez(file_name, mean_value)

class CovComputer(StatComputer):

    def __init__(self, values, mask, mean):
        self.values = values
        self.mask = mask
        self.mean = mean

        self.mean_sub = values - mean
        self.batch_values = tf.reshape(self.mean_sub, [-1, tf.shape(values)[-1]])
        self.batch_mask = tf.reshape(mask, [-1])
        self.chol, self.chol_update = cholesky_update(self.batch_values, self.batch_mask, init=float(1.0))
    
    def get_update_op(self):
        return self.chol_update
    
    def save_variable(self, sess, file_name):
        def inv_fn(chol_mat):
            cov = tf.matmul(tf.transpose(chol_mat,[0,2,1]),chol_mat)
            inv_cov = tf.linalg.inv(cov)
            return inv_cov
        target_shape = self.values.get_shape().as_list()
        num_split = target_shape[0]
        chol_list = tf.split(self.chol, num_split, 0)
        inv_list = [inv_fn(c) for c in chol_list]
        class_cov_inv = np.concatenate([sess.run(i) for i in inv_list])
        class_cov_inv = np.mean(np.reshape(class_cov_inv, target_shape + [target_shape[-1]]), 0, keepdims=True)
        
        if np.isnan(class_cov_inv).any():
            print("nan time")
            import pdb; pdb.set_trace()
        print("saving to", file_name)
        np.savez(file_name, class_cov_inv)

def process_annot(annot_tensor, feat, num_classes):
    one_hot = tf.one_hot(annot_tensor, num_classes)
    resized = tf.expand_dims(tf.image.resize_nearest_neighbor(one_hot, feat.get_shape().as_list()[1:-1]), -1)
    sorted_feats = tf.expand_dims(feat, -2)*resized #broadcast
    avg_mask = tf.cast(tf.not_equal(resized, 0), tf.float32)
    return avg_mask, sorted_feats

def run_inference_graph(model, trained_checkpoint_prefix,
                        input_dict, num_images, input_shape, pad_to_shape,
                        label_color_map, output_directory, num_classes, patch_size):
    assert len(input_shape) == 3, "input shape must be rank 3"
    effective_shape = [None] + input_shape

    if isinstance(model._feature_extractor, resnet_ex_class):
        batch = 2
    elif isinstance(model._feature_extractor, mobilenet_ex_class):
        batch = 2

    input_queue = create_input(input_dict, batch, 15, 15, 15)
    input_dict = input_queue.dequeue()

    input_tensor = input_dict[dataset_builder._IMAGE_FIELD]
    annot_tensor = input_dict[dataset_builder._LABEL_FIELD]

    input_tensor = tf.concat([input_tensor, tf.image.flip_left_right(input_tensor)], 0)
    annot_tensor = tf.concat([annot_tensor[...,0], tf.image.flip_left_right(annot_tensor)[...,0]], 0)

    outputs, placeholder_tensor = deploy_segmentation_inference_graph(
        model=model,
        input_shape=effective_shape,
        input=input_tensor,
        pad_to_shape=pad_to_shape,
        label_color_map=label_color_map)

    stats_dir = os.path.join(output_directory, "stats")
    class_mean_file = os.path.join(stats_dir, "class_mean.npz")
    class_cov_file = os.path.join(stats_dir, "class_cov_inv.npz")

    first_pass = True
    avg_mask, sorted_feats = process_annot(annot_tensor, outputs[model.final_logits_key], num_classes)

    # if os.path.exists(mean_file) and os.path.exists(class_mean_file):
    if os.path.exists(class_mean_file):
        class_mean_v = np.load(class_mean_file)["arr_0"]
        first_pass = False
        stat_computer = CovComputer(sorted_feats, avg_mask, class_mean_v)
        output_file = class_cov_file
        print("second_pass")
    else:
        stat_computer = MeanComputer(sorted_feats, avg_mask)
        output_file = class_mean_file
        print("first_pass")

    update_op = stat_computer.get_update_op()

    coord = tf.train.Coordinator()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    with tf.Session(config=config) as sess:
        saver = tf.train.Saver(tf.global_variables())

        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        saver.restore(sess, trained_checkpoint_prefix)

        k = None
        class_k = None

        num_step = num_images // batch
        for idx in range(num_step):
            start_time = timeit.default_timer()

            sess.run(update_op)

            elapsed = timeit.default_timer() - start_time
            end = "\r"
            if idx % 50 == 0:
                #every now and then do regular print
                end = "\n"
            print('{0:.4f} wall time: {1}'.format(elapsed/batch, (idx+1)*batch), end=end)
        print('{0:.4f} wall time: {1}'.format(elapsed/batch, (idx+1)*batch))
        os.makedirs(stats_dir, exist_ok=True)

        stat_computer.save_variable(sess, output_file)

        coord.request_stop()
        coord.join(threads)


def main(_):
    assert FLAGS.output_dir, '`output_dir` missing.'

    output_directory = FLAGS.output_dir
    tf.gfile.MakeDirs(output_directory)
    pipeline_config = pipeline_pb2.PipelineConfig()
    with tf.gfile.GFile(FLAGS.config_path, 'r') as f:
        text_format.Merge(f.read(), pipeline_config)

    pad_to_shape = None
    if FLAGS.input_shape:
        input_shape = [
            int(dim) if dim != '-1' else None
                for dim in FLAGS.input_shape.split(',')]
    else:
        raise ValueError('Must supply `input_shape`')

    patch_size = None
    if FLAGS.patch_size:
        patch_size = [int(dim) for dim in FLAGS.patch_size.split(',')]
        assert len(patch_size) == 2, "patch size must be h,w"

    if FLAGS.pad_to_shape:
        pad_to_shape = [
            int(dim) if dim != '-1' else None
                for dim in FLAGS.pad_to_shape.split(',')]

    label_map = (CITYSCAPES_LABEL_IDS
        if FLAGS.label_ids else CITYSCAPES_LABEL_COLORS)

    num_classes, segmentation_model = model_builder.build(
        pipeline_config.model, is_training=False)

    #input_reader = pipeline_config.eval_input_reader
    input_reader = pipeline_config.train_input_reader
    input_reader.shuffle = False
    input_reader.num_epochs = 1
    input_dict = dataset_builder.build(input_reader)

    run_inference_graph(segmentation_model, FLAGS.trained_checkpoint,
                        input_dict, input_reader.num_examples, input_shape, pad_to_shape,
                        label_map, output_directory, num_classes, patch_size)

if __name__ == '__main__':
    tf.app.run()
