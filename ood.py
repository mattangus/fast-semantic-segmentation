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
from numpy.core.umath_tests import inner1d
import matplotlib.pyplot as plt
import cv2

from protos import pipeline_pb2
from builders import model_builder
from libs.exporter import deploy_segmentation_inference_graph, _map_to_colored_labels
from libs.constants import CITYSCAPES_LABEL_COLORS, CITYSCAPES_LABEL_IDS


slim = tf.contrib.slim

flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_string('input_path', None,
                    'Path to an image or a directory of images.')

flags.DEFINE_string('input_shape', '1024,2048,3', # default Cityscapes values
                    'The shape to use for inference. This should '
                    'be in the form [height, width, channels]. A batch '
                    'dimension is not supported for this test script.')

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

flags.DEFINE_string('dist_dir', None, 'Path to write distance images')

flags.DEFINE_boolean('label_ids', False,
                     'Whether the output should be label ids.')


def _valid_file_ext(input_path):
    ext = os.path.splitext(input_path)[-1].upper()
    return ext in ['.JPG', '.JPEG', '.PNG']


def _get_images_from_path(input_path):
    image_file_paths = []
    if os.path.isdir(input_path):
        for dirpath,_,filenames in os.walk(input_path):
            for f in filenames:
                file_path = os.path.abspath(os.path.join(dirpath, f))
                if not _valid_file_ext(file_path):
                    raise ValueError('File must be JPG or PNG.')
                image_file_paths.append(file_path)
    else:
        if not _valid_file_ext(input_path):
            raise ValueError('File must be JPG or PNG.')
        image_file_paths.append(input_path)
    return image_file_paths

def nan_to_num(val):
    return tf.where(tf.is_nan(val), tf.zeros_like(val), val)

def process_logits(final_logits, mean_v, std_v, depth, pred_shape, num_classes):
    mean = tf.placeholder(tf.float32, mean_v.shape, "mean")
    std = tf.placeholder(tf.float32, std_v.shape, "std")
    in_shape = final_logits.get_shape().as_list()
    final_logits = tf.reshape(final_logits, [-1, depth])
    temp = tf.expand_dims(final_logits,-1) - mean
    temp2 = nan_to_num(1./(std + 0.000001))
    left = temp * temp2
    dist = tf.reduce_sum(tf.multiply(left, temp), 1)
    img_dist = tf.expand_dims(tf.reshape(dist, in_shape[1:-1] + [num_classes]), 0)
    full_dist = tf.image.resize_bilinear(img_dist, (pred_shape[1],pred_shape[2]))
    dist_class = tf.argmin(full_dist, -1)
    # scaled_dist = full_dist/tf.reduce_max(full_dist)
    # dist_out = (scaled_dist*255).astype(np.uint8)
    return dist_class, full_dist, mean, std #, [temp, temp2, left, dist, img_dist]

def run_inference_graph(model, trained_checkpoint_prefix,
                        input_images, input_shape, pad_to_shape,
                        label_color_map, output_directory,
                        dist_dir, num_classes):

    outputs, placeholder_tensor = deploy_segmentation_inference_graph(
        model=model,
        input_shape=input_shape,
        pad_to_shape=pad_to_shape,
        label_color_map=label_color_map)
    pred_tensor = outputs[model.main_class_predictions_key]
    final_logits = outputs[model.final_logits_key]

    mean = np.load("class_mean.npy")
    std = np.load("class_std.npy")
    other_dims = list(std.shape[-2:])
    depth = other_dims[0]

    mean = np.reshape(mean, [-1] + other_dims)
    std = np.reshape(std, [-1] + other_dims)

    dist_class, full_dist, mean_p, std_p  = process_logits(final_logits, mean, std, depth, pred_tensor.get_shape().as_list(), num_classes)
    dist_colour = _map_to_colored_labels(dist_class, pred_tensor.get_shape().as_list(), label_color_map)

    fetch = [pred_tensor, dist_colour, dist_class, full_dist]
    
    x = None
    y = None
    with tf.Session() as sess:
        input_graph_def = tf.get_default_graph().as_graph_def()
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(sess, trained_checkpoint_prefix)
        for idx, image_path in enumerate(input_images):
            image_raw = np.array(Image.open(image_path))

            start_time = timeit.default_timer()
            
            res = sess.run(fetch,
                feed_dict={placeholder_tensor: image_raw, mean_p: mean, std_p: std})
            predictions = res[0]

            # logits_output = res[1]
            # in_shape = logits_output.shape
            # logits_output = np.reshape(logits_output, [-1, depth])
            # temp = logits_output - mean
            # left = temp * np.nan_to_num(1./std)
            # dist = inner1d(left, temp)
            # img_dist = np.reshape(dist, in_shape[1:-1])
            # full_dist = cv2.resize(img_dist, (predictions.shape[2],predictions.shape[1]), interpolation=cv2.INTER_LINEAR)
            dist_out = res[1][0].astype(np.uint8)
            #import pdb; pdb.set_trace()
            # scaled_dist = full_dist/np.max(full_dist)
            # dist_out = (scaled_dist*255).astype(np.uint8)
            elapsed = timeit.default_timer() - start_time
            print('{}) wall time: {}'.format(elapsed, idx+1))
            filename = os.path.basename(image_path)
            save_location = os.path.join(output_directory, filename)
            dist_filename = os.path.join(dist_dir, filename)

            predictions = predictions.astype(np.uint8)
            output_channels = len(label_color_map[0])
            if output_channels == 1:
                predictions = np.squeeze(predictions[0],-1)
            else:
                predictions = predictions[0]
            #import pdb; pdb.set_trace()
            im = Image.fromarray(predictions)
            im.save(save_location, "PNG")

            im = Image.fromarray(dist_out)
            im.save(dist_filename, "PNG")


def main(_):
    output_directory = FLAGS.output_dir
    dist_dir = FLAGS.dist_dir
    tf.gfile.MakeDirs(output_directory)
    tf.gfile.MakeDirs(dist_dir)
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

    if FLAGS.pad_to_shape:
        pad_to_shape = [
            int(dim) if dim != '-1' else None
                for dim in FLAGS.pad_to_shape.split(',')]

    input_images = _get_images_from_path(FLAGS.input_path)
    label_map = (CITYSCAPES_LABEL_IDS
        if FLAGS.label_ids else CITYSCAPES_LABEL_COLORS)

    num_classes, segmentation_model = model_builder.build(
        pipeline_config.model, is_training=False)

    run_inference_graph(segmentation_model, FLAGS.trained_checkpoint,
                        input_images, input_shape, pad_to_shape,
                        label_map, output_directory, dist_dir, num_classes)

if __name__ == '__main__':
    tf.app.run()
