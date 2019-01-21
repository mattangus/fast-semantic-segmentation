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
from sklearn.feature_extraction import image
import copy

import extractors

from libs import sliding_window
from protos import pipeline_pb2
from builders import model_builder, dataset_builder
from libs.exporter import deploy_segmentation_inference_graph
from libs.constants import CITYSCAPES_LABEL_COLORS, CITYSCAPES_LABEL_IDS
from libs.custom_metric import streaming_mean
from submod.cholesky.cholesky_update import cholesky_update
from libs import stat_computer as stats
import feature_reader

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

flags.DEFINE_string('output_dir', None, 'Path to write outputs images.')

flags.DEFINE_boolean('label_ids', False,
                     'Whether the output should be label ids.')

flags.DEFINE_boolean('use_pool', False,
                     'avg pool over spatial dims')

flags.DEFINE_boolean('use_patch', False,
                     'avg pool over spatial dims')

flags.DEFINE_integer("max_iters", 1000000, "limit the number of iterations for large datasets")


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


def process_annot(annot_tensor, feat, num_classes):
    one_hot = tf.one_hot(annot_tensor, num_classes)
    resized = tf.expand_dims(tf.image.resize_nearest_neighbor(one_hot, feat.get_shape().as_list()[1:-1]), -1)
    sorted_feats = tf.expand_dims(feat, -2)*resized #broadcast
    avg_mask = tf.cast(tf.not_equal(resized, 0), tf.float32)
    return avg_mask, sorted_feats

def run_inference_graph(dataset, batch, num_images, input_shape, pad_to_shape,
                        label_color_map, output_directory, num_classes, patch_size):

    iterator = dataset.make_one_shot_iterator()
    next_elem = iterator.get_next()

    input_tensor = next_elem["input"]
    annot_tensor = next_elem["annot"]
    final_logits = next_elem["final_logits"]
    unscaled_logits = next_elem["unscaled_logits"]

    #can't flip with extracted features :(
    flip = False
    if flip:
        input_tensor = tf.concat([input_tensor, tf.image.flip_left_right(input_tensor)], 0)
        annot_tensor = tf.concat([annot_tensor[...,0], tf.image.flip_left_right(annot_tensor)[...,0]], 0)
    else:
        annot_tensor = annot_tensor[...,0]

    #import pdb; pdb.set_trace()

    stats_dir = os.path.join(output_directory, "stats")
    class_mean_file = os.path.join(stats_dir, "class_mean.npz")
    class_cov_file = os.path.join(stats_dir, "class_cov_inv.npz")

    first_pass = True
    avg_mask, sorted_feats = process_annot(annot_tensor, final_logits, num_classes)

    # if os.path.exists(mean_file) and os.path.exists(class_mean_file):
    if os.path.exists(class_mean_file):
        class_mean_v = np.load(class_mean_file)["arr_0"]
        first_pass = False
        if FLAGS.use_patch:
            comp = stats.PatchCovComputer
        else:
            comp = stats.CovComputer
        stat_computer = comp(sorted_feats, avg_mask, class_mean_v)
        output_file = class_cov_file
        print("second_pass")
    else:
        if FLAGS.use_patch:
            comp = stats.PatchMeanComputer
        else:
            comp = stats.MeanComputer
        stat_computer = comp(sorted_feats, avg_mask)
        output_file = class_mean_file
        print("first_pass")

    update_op = stat_computer.get_update_op()
    
    config = tf.ConfigProto()
    #config.gpu_options.allow_growth=True

    with tf.Session(config=config) as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        k = None
        class_k = None

        num_step = num_images // batch
        for idx in range(num_step):
            start_time = timeit.default_timer()

            # if (idx+1)*batch ==2888:
            #     import pdb; pdb.set_trace()
            
            try:
                sess.run(update_op)
            except tf.errors.DataLossError:
                print("caught dataloss")
                break

            elapsed = timeit.default_timer() - start_time
            end = "\r"
            if idx % 50 == 0:
                #every now and then do regular print
                end = "\n"
            print('{0:.4f}: {1}'.format(elapsed/batch, (idx+1)*batch), end=end)
        print('{0:.4f}: {1}'.format(elapsed/batch, (idx+1)*batch))
        os.makedirs(stats_dir, exist_ok=True)

        stat_computer.save_variable(sess, output_file)


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

    batch = 1

    feats_dir = os.path.join(output_directory, "feats")
    dataset_filter = os.path.join(feats_dir, "feats_train.record")
    dataset = feature_reader.get_feature_dataset(dataset_filter, batch, drop_remain=True)

    input_reader = pipeline_config.train_input_reader
    iters = min(input_reader.num_examples, FLAGS.max_iters)
    num_classes = pipeline_config.model.pspnet.num_classes

    run_inference_graph(dataset, batch, iters, input_shape, pad_to_shape,
                        label_map, output_directory, num_classes, patch_size)

if __name__ == '__main__':
    tf.app.run()
