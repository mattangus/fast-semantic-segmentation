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

flags.DEFINE_string('eval_dir', None, 'Path to write outputs images.')

flags.DEFINE_boolean('label_ids', False,
                     'Whether the output should be label ids.')

flags.DEFINE_boolean('use_class', False,
                     'Use class covariance or not')


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

def process_logits(final_logits, mean_v, var_v, depth, pred_shape, num_classes, use_class):
    mean_p = tf.placeholder(tf.float32, mean_v.shape, "mean")
    var_p = tf.placeholder(tf.float32, var_v.shape, "var")
    var = var_p

    in_shape = final_logits.get_shape().as_list() 
    if not use_class:
        var = tf.tile(tf.expand_dims(var, 3), [1, 1, 1, num_classes, 1, 1])
    var = tf.reshape(var, [-1, in_shape[-1], in_shape[-1]])
    mean = tf.reshape(mean_p, [-1, num_classes, in_shape[-1]])

    final_logits = tf.reshape(final_logits, [-1, depth])
    temp = tf.expand_dims(final_logits,-2) - mean
    temp = tf.expand_dims(tf.reshape(temp, [-1, in_shape[-1]]), 1)
    #temp2 = nan_to_num(1./(var + 0.000001))
    
    left = tf.matmul(temp, var)
    dist = tf.squeeze(tf.matmul(left, temp, transpose_b=True))

    img_dist = tf.expand_dims(tf.reshape(dist, in_shape[1:-1] + [num_classes]), 0)
    full_dist = tf.image.resize_bilinear(img_dist, (pred_shape[1],pred_shape[2]))
    dist_class = tf.argmin(full_dist, -1)
    # scaled_dist = full_dist/tf.reduce_max(full_dist)
    # dist_out = (scaled_dist*255).astype(np.uint8)
    return dist_class, full_dist, mean_p, var_p #, [temp, temp2, left, dist, img_dist]

def run_inference_graph(model, trained_checkpoint_prefix,
                        input_images, input_shape, pad_to_shape,
                        label_color_map, output_directory,
                        dist_dir, eval_dir, num_classes):

    outputs, placeholder_tensor = deploy_segmentation_inference_graph(
        model=model,
        input_shape=input_shape,
        pad_to_shape=pad_to_shape,
        label_color_map=label_color_map)
    pred_tensor = outputs[model.main_class_predictions_key]
    final_logits = outputs[model.final_logits_key]

    use_class = FLAGS.use_class

    stats_dir = os.path.join(eval_dir, "stats")
    mean_file = os.path.join(stats_dir, "mean.npz")
    class_mean_file = os.path.join(stats_dir, "class_mean.npz")
    cov_file = os.path.join(stats_dir, "cov_inv.npz")
    class_cov_file = os.path.join(stats_dir, "class_cov_inv.npz")
    if use_class:
        cov_file = class_cov_file
    print("loading means and covs")
    mean = np.load(class_mean_file)["arr_0"]
    var = np.load(cov_file)["arr_0"]
    print("done loading")
    var_dims = list(var.shape[-2:])
    mean_dims = list(mean.shape[-2:])
    depth = mean_dims[-1]
    
    #mean = np.reshape(mean, [-1] + mean_dims)
    #var = np.reshape(var, [-1] + var_dims)
    
    dist_class, full_dist, mean_p, var_p  = process_logits(final_logits, mean, var, depth, pred_tensor.get_shape().as_list(), num_classes, use_class)
    dist_colour = _map_to_colored_labels(dist_class, pred_tensor.get_shape().as_list(), label_color_map)

    mean = np.reshape(mean, mean_p.get_shape().as_list())
    var = np.reshape(var, var_p.get_shape().as_list())

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
                feed_dict={placeholder_tensor: image_raw, mean_p: mean, var_p: var})
            predictions = res[0]

            # logits_output = res[1]
            # in_shape = logits_output.shape
            # logits_output = np.reshape(logits_output, [-1, depth])
            # temp = logits_output - mean
            # left = temp * np.nan_to_num(1./var)
            # dist = inner1d(left, temp)
            # img_dist = np.reshape(dist, in_shape[1:-1])
            # full_dist = cv2.resize(img_dist, (predictions.shape[2],predictions.shape[1]), interpolation=cv2.INTER_LINEAR)
            dist_out = res[1][0].astype(np.uint8)
            # full_dist_out = res[3][0]
            # for i in range(num_classes):
            #     temp = full_dist_out[:,:,i]
            #     cv2.imshow(str(i), temp/np.max(temp))
            # cv2.waitKey()
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
    eval_dir = FLAGS.eval_dir
    output_directory = os.path.join(eval_dir, "inf")
    if FLAGS.use_class:
        dist_dir = os.path.join(eval_dir, "class_dist")
    else:
        dist_dir = os.path.join(eval_dir, "dist")
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
                        label_map, output_directory, dist_dir, eval_dir, num_classes)

if __name__ == '__main__':
    tf.app.run()
