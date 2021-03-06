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
from protos.config_reader import read_config

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

flags.DEFINE_string('model_config', None,
                    'Path to a pipeline_pb2.TrainEvalConfig config '
                    'file. If provided, other configs are ignored')
flags.mark_flag_as_required('model_config')

flags.DEFINE_string('data_config', None,
                    'Path to a pipeline_pb2.TrainEvalConfig config '
                    'file. If provided, other configs are ignored')
flags.mark_flag_as_required('data_config')


flags.DEFINE_string('trained_checkpoint', None,
                    'Path to trained checkpoint, typically of the form '
                    'path/to/model.ckpt')

flags.DEFINE_string('output_dir', None, 'Path to write outputs images.')

flags.DEFINE_boolean('label_ids', False,
                     'Whether the output should be label ids.')

flags.DEFINE_boolean('use_pool', False,
                     'avg pool over spatial dims')

flags.DEFINE_boolean('use_patch', False,
                     'avg pool over spatial dims')

flags.DEFINE_boolean('use_dtform', False,
                     'use distance transform')

flags.DEFINE_integer("max_iters", 1000000000, "limit the number of iterations for large datasets")

epoch = 1
def create_input(dataset,
                          batch_size):
    
    def cast_and_reshape(tensor_dict, dicy_key):
        items = tensor_dict[dicy_key]
        float_images = tf.to_float(items)
        tensor_dict[dicy_key] = float_images
        return tensor_dict

    dataset = dataset.map(lambda x: cast_and_reshape(x,
                                    dataset_builder._IMAGE_FIELD))

    dataset = dataset.prefetch(batch_size)

    dataset = dataset.batch(batch_size, drop_remainder=True)
    #dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))

    return dataset

# def create_input(tensor_dict,
#                 batch_size,
#                 batch_queue_capacity,
#                 batch_queue_threads,
#                 prefetch_queue_capacity):

#     def cast_and_reshape(tensor_dict, dicy_key):
#         import pdb; pdb.set_trace()
#         items = tensor_dict[dicy_key]
#         float_images = tf.to_float(items)
#         tensor_dict[dicy_key] = float_images
#         return tensor_dict

#     tensor_dict = cast_and_reshape(tensor_dict,
#                     dataset_builder._IMAGE_FIELD)

#     batched_tensors = tf.train.batch(tensor_dict,
#         batch_size=batch_size, num_threads=batch_queue_threads,
#         capacity=batch_queue_capacity, dynamic_pad=True,
#         allow_smaller_final_batch=False)

#     return prefetch_queue.prefetch_queue(batched_tensors,
#         capacity=prefetch_queue_capacity,
#         dynamic_pad=False)

def dist_transform(annot):
    lr = tf.not_equal(annot[:,1:,1:] - annot[:,:-1,1:],0)
    ud = tf.not_equal(annot[:,1:,1:] - annot[:,1:,:-1],0)
    both = tf.to_float(tf.logical_not(tf.logical_or(lr,ud)))*255
    both = tf.pad(both,[[0,0],[0,1],[0,1]])

    def tform(img):
        ret = cv2.distanceTransform(img, cv2.DIST_L2, 3)
        return ret

    dtform = tf.map_fn(lambda x: tf.py_func(tform, [tf.cast(x,tf.uint8)], tf.float32, False), both)
    dtform.set_shape(both.shape)
    return dtform

def process_annot(annot_tensor, feat, num_classes):
    ed = tf.expand_dims
    one_hot = tf.one_hot(annot_tensor, num_classes)
    resized = ed(tf.image.resize_nearest_neighbor(one_hot, feat.get_shape().as_list()[1:-1]), -1)
    sorted_feats = ed(feat, -2)*resized #broadcast
    avg_mask = tf.cast(tf.not_equal(resized, 0), tf.float32)
    if FLAGS.use_dtform:
        max_dist = 10
        dtform = dist_transform(annot_tensor)
        mask = tf.to_float(dtform > max_dist)
        weights = (max_dist*mask + (1-mask)*dtform)/max_dist
        weight_resize = ed(tf.image.resize_nearest_neighbor(ed(weights, -1), feat.get_shape().as_list()[1:-1]),-1)
        avg_mask = avg_mask*weight_resize
    return avg_mask, sorted_feats

def run_inference_graph(model, trained_checkpoint_prefix,
                        input_dict, num_images, input_shape, pad_to_shape,
                        label_color_map, output_directory, num_classes, patch_size):
    assert len(input_shape) == 3, "input shape must be rank 3"
    effective_shape = [None] + input_shape

    batch = 1
    if isinstance(model._feature_extractor, resnet_ex_class):
        batch = 2
    elif isinstance(model._feature_extractor, mobilenet_ex_class):
        batch = 1

    dataset = create_input(input_dict, batch)
    dataset = dataset.apply(tf.data.experimental.ignore_errors())
    data_iterator = dataset.make_one_shot_iterator()
    input_dict = data_iterator.get_next()

    input_tensor = input_dict[dataset_builder._IMAGE_FIELD]
    annot_tensor = input_dict[dataset_builder._LABEL_FIELD]

    flip = True
    if flip:
        input_tensor = tf.concat([input_tensor, tf.image.flip_left_right(input_tensor)], 0)
        annot_tensor = tf.concat([annot_tensor[...,0], tf.image.flip_left_right(annot_tensor)[...,0]], 0)
    else:
        annot_tensor = annot_tensor[...,0]

    #import pdb; pdb.set_trace()

    outputs, placeholder_tensor = deploy_segmentation_inference_graph(
        model=model,
        input_shape=effective_shape,
        input=input_tensor,
        pad_to_shape=pad_to_shape,
        label_color_map=label_color_map)

    final_logits = outputs[model.final_logits_key]

    if FLAGS.use_dtform:
        stats_dir = os.path.join(output_directory, "stats.dtform")
    else:
        stats_dir = os.path.join(output_directory, "stats")
    class_mean_file = os.path.join(stats_dir, "class_mean.npz")
    class_cov_file = os.path.join(stats_dir, "class_cov_inv.npz")

    first_pass = True
    with tf.device("gpu:1"):
        avg_mask, sorted_feats = process_annot(annot_tensor, final_logits, num_classes)

        # if os.path.exists(mean_file) and os.path.exists(class_mean_file):
        feed_dict = {}
        if os.path.exists(class_mean_file):
            class_mean_v = np.load(class_mean_file)["arr_0"]

            # class_mean_pl = tf.placeholder(tf.float32, class_mean_v.shape)
            # feed_dict[class_mean_pl] = class_mean_v
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
    
    coord = tf.train.Coordinator()
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth=True

    full_eye = None
    coord = tf.train.Coordinator()

    with tf.Session(config=config) as sess:
        saver = tf.train.Saver(tf.global_variables())

        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        saver.restore(sess, trained_checkpoint_prefix)

        num_step = epoch * num_images // batch
        print("running for", num_step*batch)
        for idx in range(num_step):
            start_time = timeit.default_timer()

            sess.run(update_op, feed_dict=feed_dict)

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
    pipeline_config = read_config(FLAGS.model_config, FLAGS.data_config)

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

    ignore_label = pipeline_config.input_reader.ignore_label

    num_classes, segmentation_model = model_builder.build(
        pipeline_config.model, ignore_label=ignore_label, is_training=False)

    #input_reader = pipeline_config.eval_input_reader
    input_reader = pipeline_config.input_reader
    input_reader.shuffle = False
    input_dict = dataset_builder.build(input_reader, epoch)

    num_examples = sum([r.num_examples for r in input_reader.tf_record_input_reader])
    iters = min(num_examples, FLAGS.max_iters)

    run_inference_graph(segmentation_model, FLAGS.trained_checkpoint,
                        input_dict, iters, input_shape, pad_to_shape,
                        label_map, output_directory, num_classes, patch_size)

if __name__ == '__main__':
    tf.app.run()
