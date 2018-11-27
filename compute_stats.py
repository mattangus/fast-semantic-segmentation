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

def _valid_file_ext(input_path):
    ext = os.path.splitext(input_path)[-1].upper()
    return ext in ['.JPG', '.JPEG', '.PNG']

def safe_div(a,b, thresh = 1e-7):
    b = tf.broadcast_to(a, tf.shape(b)) #broadcast
    return tf.where(tf.less(tf.abs(b), thresh), b, a/b)

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

def p_inv(matrix):
    
    """Returns the Moore-Penrose pseudo-inverse"""
    import pdb; pdb.set_trace()
    s, u, v = tf.svd(matrix)
    
    s_inv = tf.matrix_diag(safe_div(float(1),s))

    perm = list(range(len(u.get_shape())))
    perm[-1], perm[-2] = perm[-2], perm[-1]

    #v s^-1 u^T
    return tf.matmul(v, tf.matmul(s_inv, tf.transpose(u, perm)))

# def b_inv(b_mat):
#     eye = b_mat.new_ones(b_mat.size(-1)).diag().expand_as(b_mat)
#     b_inv, _ = torch.gesv(eye, b_mat)
#     return b_inv

def compute_mean(m_km1, x_k, k, mask):
    if m_km1 is None:
        return torch.tensor(x_k), torch.ones(x_k.shape)
    x_k = torch.tensor(x_k)
    mask = torch.tensor(mask)
    temp = (x_k - m_km1)*mask
    m_k = m_km1 + temp/k
    k += mask
    return m_k, k

# def compute_cov_inv(m, v_km1_inv, x_k, k, mask):
#     dim = x_k.shape[-1]
#     out_shape = list(x_k.shape[:-1]) + [dim, x_k.shape[-1]]
#     if v_km1_inv is None:
#         return torch.zeros(out_shape), torch.ones(out_shape)
#     x_k = torch.tensor(x_k)
#     mask = torch.tensor(mask)
#     mask = mask.unsqueeze(-1)
#     temp = (x_k - m)
#     sigma = torch.bmm(temp.view(-1, dim).unsqueeze(2), temp.view(-1, dim).unsqueeze(1)).view(out_shape)
#     v_k_inv = b_inv(sigma)*mask + v_km1_inv
#     k += mask
#     return v_k_inv, k

# def compute_stats(m_km1, v_km1_inv, x_k, mask, x_k_mean, k, first_pass):
#     if first_pass:
#         m_k, k = compute_mean(m_km1, x_k, k, mask)
#         v_k_inv = v_km1_inv
#     else:
#         m_k = m_km1
#         v_k_inv, k = compute_cov_inv(m_k, v_km1_inv, x_k, k, mask)
#     return m_k, v_k_inv, k

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

    x = None
    y = None
    class_m_k = None
    class_v_k_inv = None
    first_pass = True

    # if os.path.exists(mean_file) and os.path.exists(class_mean_file):
    if os.path.exists(class_mean_file):
        class_m_k = np.load(class_mean_file)["arr_0"]
        first_pass = False

    avg_mask, sorted_feats = process_annot(annot_tensor, outputs[model.final_logits_key], num_classes)
    fetch = [sorted_feats, avg_mask]

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    with tf.Session(config=config) as sess:
        saver = tf.train.Saver(tf.global_variables())

        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        tf.train.start_queue_runners(sess)
        saver.restore(sess, trained_checkpoint_prefix)

        k = None
        class_k = None
        if first_pass:
            passes = [True, False]
            cache = None
            #cache is too large. need to try mmap
            #cache = []
        else:
            passes = [False] #means loaded from disk
            cache = None

        num_step = num_images // batch

        for first_pass in passes:
            if first_pass:
                print("first pass")
            else:
                print("second pass")
                mean_sub = sorted_feats - class_m_k
                batch_feats = tf.reshape(mean_sub, [-1, tf.shape(sorted_feats)[-1]])
                batch_mask = tf.reshape(avg_mask, [-1])
                chol, chol_update = cholesky_update(batch_feats, batch_mask)
                sess.run(tf.global_variables_initializer())
            for idx in range(num_step):
                start_time = timeit.default_timer()

                #cache results for second pass
                if first_pass:
                    res = sess.run(fetch)

                    sorted_logits = res[0]
                    mask = res[1]
                    mean_logits = sorted_logits#res[2]

                    for b in range(sorted_logits.shape[0]): #should only be 1
                        # # ret = compute_stats(class_m_k, class_v_k_inv,
                        #                     sorted_logits[b:b+1], mask[b:b+1],
                        #                     mean_logits[b:b+1], class_k, first_pass)
                        class_m_k, class_k = compute_mean(class_m_k, sorted_logits[b:b+1], class_k, mask[b:b+1])
                        # class_m_k, class_k = ret
                else:
                    sess.run(chol_update)
                    #cur_chol = sess.run(chol)

                    # import pdb; pdb.set_trace()

                # if idx > 10:
                #     import pdb; pdb.set_trace()
                # if idx*batch > 50:
                #     break

                elapsed = timeit.default_timer() - start_time
                print('{}) wall time: {}'.format(elapsed/batch, (idx+1)*batch))

            os.makedirs(stats_dir, exist_ok=True)

            if first_pass:
                class_m_k_np = class_m_k.numpy()
                if np.isnan(class_m_k_np).any():
                    print("nan time")
                    import pdb; pdb.set_trace()
                print("saving to", class_mean_file)
                np.savez(class_mean_file, class_m_k_np)
            else:
                #cur_chol = sess.run(chol)
                def inv_fn(chol_mat):
                    cov = tf.matmul(chol_mat,tf.transpose(chol_mat,[0,2,1]))
                    inv_cov = tf.linalg.inv(cov)
                    return inv_cov
                num_split = batch*2
                chol_list = tf.split(chol, num_split, 0)
                inv_list = [inv_fn(c) for c in chol_list]
                class_cov_inv = np.concatenate([sess.run(i) for i in inv_list])
                target_shape = sorted_feats.get_shape().as_list()
                class_cov_inv = np.mean(np.reshape(class_cov_inv, target_shape + [target_shape[-1]]), 0, keepdims=True)
                #inv_cov = tfp.math.pinv(cov)
                # try:
                #     class_cov_inv = sess.run(inv_cov)
                # except Exception as ex:
                #     import pdb; pdb.set_trace()
                
                if np.isnan(class_cov_inv).any():
                    print("nan time")
                    import pdb; pdb.set_trace()
                print("saving to", class_cov_file)
                np.savez(class_cov_file, class_cov_inv)
        # print(save_location)
        # res = 25
        # vec_pred = np.fliplr(vec_pred)
        # plt.quiver(vec_pred[0,::res,::res,0], vec_pred[0,::res,::res,1])
        # plt.show()


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

    input_reader = pipeline_config.train_input_reader
    #input_reader = pipeline_config.eval_input_reader # for testing
    input_reader.shuffle = False
    input_reader.num_epochs = 1
    input_dict = dataset_builder.build(input_reader)

    run_inference_graph(segmentation_model, FLAGS.trained_checkpoint,
                        input_dict, input_reader.num_examples, input_shape, pad_to_shape,
                        label_map, output_directory, num_classes, patch_size)

if __name__ == '__main__':
    tf.app.run()
