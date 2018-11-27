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
from builders import model_builder, dataset_builder
from libs.exporter import deploy_segmentation_inference_graph, _map_to_colored_labels
from libs.constants import CITYSCAPES_LABEL_COLORS, CITYSCAPES_LABEL_IDS, URSA_LABEL_COLORS
from libs.metrics import mean_iou

slim = tf.contrib.slim

flags = tf.app.flags

FLAGS = flags.FLAGS

prefetch_queue = slim.prefetch_queue

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

flags.DEFINE_boolean('use_pool', False,
                     '')

flags.DEFINE_boolean('write_out', False,
                     '')

                     

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
                    print("invalid path: '" + file_path + "'. skipping")
                    #raise ValueError('File must be JPG or PNG.')
                else:
                    image_file_paths.append(file_path)
    else:
        if not _valid_file_ext(input_path):
            raise ValueError('File must be JPG or PNG.')
        image_file_paths.append(input_path)
    return image_file_paths

def nan_to_num(val):
    return tf.where(tf.is_nan(val), tf.zeros_like(val), val)

# def process_logits_l2(final_logits, mean_v, var_v, depth, pred_shape, num_classes, use_pool):
#     print("WARNING: Using l2 norm. not mahalanobis distance")
#     mean_p = tf.placeholder(tf.float32, mean_v.shape, "mean")
#     var_inv_p = tf.placeholder(tf.float32, var_v.shape, "var_inv")
#     var_inv = var_inv_p
#     mean = mean_p

#     if use_pool:
#         var_brod = tf.ones_like(var_inv)
#         mean_brod = tf.ones_like(mean)
#         #import pdb; pdb.set_trace()
#         var_inv = tf.reduce_mean(var_inv, axis=[0,1,2], keepdims=True)*var_brod
#         mean = tf.reduce_mean(mean, axis=[0,1,2], keepdims=True)*mean_brod

#     in_shape = final_logits.get_shape().as_list() 
#     var_inv = tf.reshape(var_inv, [-1, in_shape[-1], in_shape[-1]])
#     mean = tf.reshape(mean, [-1, num_classes, in_shape[-1]])

#     final_logits = tf.reshape(final_logits, [-1, depth])
    
#     mean_sub = tf.expand_dims(final_logits,-2) - mean

#     dist = tf.reduce_sum(tf.square(mean_sub),1)

#     img_dist = tf.expand_dims(tf.reshape(dist, in_shape[1:-1] + [num_classes]), 0)
#     img_dist = tf.where(tf.equal(img_dist, tf.zeros_like(img_dist)), tf.ones_like(img_dist)*float("inf"), img_dist)
#     full_dist = tf.image.resize_bilinear(img_dist, (pred_shape[1],pred_shape[2]))
#     dist_class = tf.argmin(full_dist, -1)
#     min_dist = tf.reduce_min(full_dist, -1)
#     # scaled_dist = full_dist/tf.reduce_max(full_dist)
#     # dist_out = (scaled_dist*255).astype(np.uint8)
#     return dist_class, full_dist, min_dist, mean_p, var_inv_p #, [temp, temp2, left, dist, img_dist]

def process_logits(final_logits, mean_v, var_inv_v, depth, pred_shape, num_classes, use_pool):
    mean_p = tf.placeholder(tf.float32, mean_v.shape, "mean")
    var_inv_p = tf.placeholder(tf.float32, var_inv_v.shape, "var_inv")
    var_inv = var_inv_p
    mean = mean_p

    if use_pool:
        var_brod = tf.ones_like(var_inv)
        mean_brod = tf.ones_like(mean)
        # import pdb; pdb.set_trace()
        var_inv = tf.reduce_sum(var_inv, axis=[0,1,2], keepdims=True)*var_brod
        #mean = tf.reduce_mean(mean, axis=[0,1,2], keepdims=True)*mean_brod

    #import pdb; pdb.set_trace()

    in_shape = final_logits.get_shape().as_list() 
    var_inv = tf.reshape(var_inv, [-1, in_shape[-1], in_shape[-1]])
    mean = tf.reshape(mean, [-1, num_classes, in_shape[-1]])

    final_logits = tf.reshape(final_logits, [-1, depth])
    
    mean_sub = tf.expand_dims(final_logits,-2) - mean
    mean_sub = tf.expand_dims(tf.reshape(mean_sub, [-1, in_shape[-1]]), 1)

    #var_inv = tf.tile(var_inv,[np.prod(in_shape[1:3]), 1, 1])
    left = tf.matmul(mean_sub, var_inv)
    dist = tf.squeeze(tf.matmul(left, mean_sub, transpose_b=True))

    img_dist = tf.expand_dims(tf.reshape(dist, in_shape[1:-1] + [num_classes]), 0)
    img_dist = tf.where(tf.equal(img_dist, tf.zeros_like(img_dist)), tf.ones_like(img_dist)*float("inf"), img_dist)
    full_dist = tf.image.resize_bilinear(img_dist, (pred_shape[1],pred_shape[2]))
    dist_class = tf.expand_dims(tf.argmin(full_dist, -1),-1)
    min_dist = tf.reduce_min(full_dist, -1)
    # scaled_dist = full_dist/tf.reduce_max(full_dist)
    # dist_out = (scaled_dist*255).astype(np.uint8)
    return dist_class, full_dist, min_dist, mean_p, var_inv_p #, [temp, temp2, left, dist, img_dist]

def get_miou(labels,
             predictions,
             num_classes,
             ignore_label):
    ne = [tf.not_equal(labels, il) for il in ignore_label]
    neg_validity_mask = ne.pop(0)
    for v in ne:
        neg_validity_mask = tf.logical_and(neg_validity_mask, v)
    eval_labels = tf.where(neg_validity_mask, labels,
                            tf.zeros_like(labels))

    return mean_iou(eval_labels, predictions, num_classes)

def run_inference_graph(model, trained_checkpoint_prefix,
                        input_dict, num_images, ignore_label, input_shape, pad_to_shape,
                        label_color_map, output_directory, num_classes, eval_dir,
                        min_dir, dist_dir):
    assert len(input_shape) == 3, "input shape must be rank 3"
    effective_shape = [None] + input_shape
    batch = 1

    input_queue = create_input(input_dict, batch, 15, 15, 15)
    input_dict = input_queue.dequeue()

    input_tensor = input_dict[dataset_builder._IMAGE_FIELD]
    annot_tensor = input_dict[dataset_builder._LABEL_FIELD]
    input_name = input_dict[dataset_builder._IMAGE_NAME_FIELD]
    
    outputs, placeholder_tensor = deploy_segmentation_inference_graph(
        model=model,
        input_shape=effective_shape,
        input=input_tensor,
        pad_to_shape=pad_to_shape)
    
    pred_tensor = outputs[model.main_class_predictions_key]
    final_logits = outputs[model.final_logits_key]

    stats_dir = os.path.join(eval_dir, "stats")
    class_mean_file = os.path.join(stats_dir, "class_mean.npz")
    class_cov_file = os.path.join(stats_dir, "class_cov_inv.npz")

    use_pool = FLAGS.use_pool

    print("loading means and covs")
    mean = np.load(class_mean_file)["arr_0"]
    var_inv = np.load(class_cov_file)["arr_0"]
    print("done loading")
    var_dims = list(var_inv.shape[-2:])
    mean_dims = list(mean.shape[-2:])
    depth = mean_dims[-1]
    
    #mean = np.reshape(mean, [-1] + mean_dims)
    #var_inv = np.reshape(var_inv, [-1] + var_dims)
    
    dist_class, full_dist, min_dist, mean_p, var_inv_p  = process_logits(final_logits, mean, var_inv, depth, pred_tensor.get_shape().as_list(), num_classes, use_pool)
    dist_colour = _map_to_colored_labels(dist_class, pred_tensor.get_shape().as_list(), label_color_map)
    pred_colour = _map_to_colored_labels(pred_tensor, pred_tensor.get_shape().as_list(), label_color_map)

    with tf.variable_scope("PredIou"):
        pred_miou, pred_conf_mat, pred_update = get_miou(annot_tensor, pred_tensor, num_classes, ignore_label)
    with tf.variable_scope("DistIou"):
        dist_miou, dist_conf_mat, dist_update = get_miou(annot_tensor, dist_class, num_classes, ignore_label)

    iou_update = tf.group([pred_update, dist_update])

    mean = np.reshape(mean, mean_p.get_shape().as_list())
    var_inv = np.reshape(var_inv, var_inv_p.get_shape().as_list())

    fetch = [input_name, pred_tensor, pred_colour, dist_colour, dist_class, full_dist, min_dist, iou_update]
    
    num_step = num_images // batch

    x = None
    y = None
    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        tf.train.start_queue_runners(sess)
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(sess, trained_checkpoint_prefix)
        for idx in range(num_step):

            start_time = timeit.default_timer()
            
            res = sess.run(fetch,
                feed_dict={mean_p: mean, var_inv_p: var_inv})
            image_path = res[0][0].decode("utf-8")

            pred_miou_v, dist_miou_v = sess.run([pred_miou, dist_miou])
            #import pdb; pdb.set_trace()
            #full_dist_out = full_dist_out/np.nanmax(full_dist_out)

            for i in range(num_classes):
                temp = full_dist_out[:,:,i]
                temp[np.logical_not(np.isfinite(temp))] = 0
                temp = temp/np.max(temp)
                cv2.imshow(str(i), temp)
            cv2.waitKey()
            # import pdb; pdb.set_trace()
            # scaled_dist = full_dist/np.max(full_dist)
            # dist_out = (scaled_dist*255).astype(np.uint8)
            elapsed = timeit.default_timer() - start_time
            end = "\r"
            if idx % 50 == 0:
                #every now and then do regular print
                end = "\n"
            print('{0:.4f} iter: {1}, pred iou: {2:.6f}, dist iou: {3:.6f}'.format(elapsed, idx+1, pred_miou_v, dist_miou_v), end=end)

            if FLAGS.write_out:
                predictions = res[1]
                prediction_colour = res[2]
                
                dist_out = res[3][0].astype(np.uint8)
                full_dist_out = res[5][0]
                min_dist_out = res[6][0]

                min_dist = np.expand_dims(np.nanmin(full_dist_out, -1), -1)
                min_dist[np.logical_not(np.isfinite(min_dist))] = 0
                min_dist = (255*min_dist/np.max(min_dist)).astype(np.uint8)

                filename = os.path.basename(image_path)
                save_location = os.path.join(output_directory, filename)
                dist_filename = os.path.join(dist_dir, filename)
                min_filename = os.path.join(min_dir, filename)

                prediction_colour = prediction_colour.astype(np.uint8)
                output_channels = len(label_color_map[0])
                if output_channels == 1:
                    prediction_colour = np.squeeze(prediction_colour[0],-1)
                else:
                    prediction_colour = prediction_colour[0]
                #import pdb; pdb.set_trace()
                cv2.imwrite(save_location, prediction_colour)
                cv2.imwrite(min_filename, min_dist)
                cv2.imwrite(dist_filename, dist_out)
        print('{0:.4f} iter: {1}, pred iou: {2:.6f}, dist iou: {3:.6f}'.format(elapsed, idx+1, pred_miou_v, dist_miou_v))


def main(_):
    eval_dir = FLAGS.output_dir
    output_directory = os.path.join(eval_dir, "inf")
    suff = ""
    if FLAGS.use_pool:
        suff = "_pool"
    dist_dir = os.path.join(eval_dir, "class_dist" + suff)
    min_dir = os.path.join(eval_dir, "min" + suff)
    tf.gfile.MakeDirs(output_directory)
    tf.gfile.MakeDirs(min_dir)
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

    label_map = (CITYSCAPES_LABEL_IDS
        if FLAGS.label_ids else URSA_LABEL_COLORS)

    num_classes, segmentation_model = model_builder.build(
        pipeline_config.model, is_training=False)

    input_reader = pipeline_config.eval_input_reader 
    input_reader.shuffle = False
    input_reader.num_epochs = 1
    input_dict = dataset_builder.build(input_reader)

    ignore_label = pipeline_config.eval_config.ignore_label

    run_inference_graph(segmentation_model, FLAGS.trained_checkpoint,
                        input_dict, input_reader.num_examples, ignore_label, input_shape, pad_to_shape,
                        label_map, output_directory, num_classes, eval_dir, min_dir, dist_dir)

if __name__ == '__main__':
    tf.app.run()
