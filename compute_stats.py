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

import extractors

from libs import sliding_window
from protos import pipeline_pb2
from builders import model_builder, dataset_builder
from libs.exporter import deploy_segmentation_inference_graph
from libs.constants import CITYSCAPES_LABEL_COLORS, CITYSCAPES_LABEL_IDS
from libs.custom_metric import streaming_mean

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

flags.DEFINE_boolean('use_pool', False,
                     'avg pool over spatial dims')

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
                    raise ValueError('File must be JPG or PNG.')
                image_file_paths.append(file_path)
    else:
        if not _valid_file_ext(input_path):
            raise ValueError('File must be JPG or PNG.')
        image_file_paths.append(input_path)
    return image_file_paths

def b_inv(b_mat):
    eye = b_mat.new_ones(b_mat.size(-1)).diag().expand_as(b_mat)
    b_inv, _ = torch.gesv(eye, b_mat)
    return b_inv

def compute_mean(m_km1, x_k, k, mask):
    if m_km1 is None:
        return torch.tensor(x_k).cuda(), torch.ones(x_k.shape).cuda()
    x_k = torch.tensor(x_k).cuda()
    mask = torch.tensor(mask).cuda()
    temp = (x_k - m_km1)*mask
    m_k = m_km1 + temp/k
    k += mask
    return m_k, k

# def compute_cov_inv(m, v_km1_inv, x_k, k, mask):
#     dim = x_k.shape[-1]
#     out_shape = list(x_k.shape[:-1]) + [dim, x_k.shape[-1]]
#     if v_km1_inv is None:
#         return torch.zeros(out_shape).cuda(), torch.ones(out_shape).cuda()
#     x_k = torch.tensor(x_k).cuda()
#     mask = torch.tensor(mask).cuda()
#     mask = mask.unsqueeze(-1)
#     temp = (x_k - m)
#     sigma = torch.bmm(temp.view(-1, dim).unsqueeze(2), temp.view(-1, dim).unsqueeze(1)).view(out_shape)
#     # try:
#     sig_inv = torch.inverse(sigma)
#     eye_test = torch.bmm(sig_inv[0,0,0], sigma[0,0,0])
#     v_k_inv = sig_inv*mask + v_km1_inv
#     # except Exception as ex:
#     #     import pdb; pdb.set_trace()
#     #     print("here")
#     k += mask
#     import pdb; pdb.set_trace()
#     return v_k_inv, k

def compute_cov_inv(m, v_km1_inv, sig_inv_k, k, mask):
    out_shape = sig_inv_k.shape
    if v_km1_inv is None:
        return torch.zeros(out_shape).cuda(), torch.ones(out_shape).cuda()
    sig_inv_k = torch.tensor(sig_inv_k).cuda()
    v_k_inv = sig_inv_k + v_km1_inv
    k += mask
    return v_k_inv, k

def compute_stats(m_km1, v_km1_inv, x_k, mask, sig, k, first_pass):
    if first_pass:
        m_k, k = compute_mean(m_km1, x_k, k, mask)
        v_k_inv = v_km1_inv
    else:
        m_k = m_km1
        v_k_inv, k = compute_mean(v_km1_inv, sig, k, np.expand_dims(mask, -1))
    return m_k, v_k_inv, k

def safe_div(a,b):
    b = tf.ones_like(a)*b #broadcast
    return tf.where(tf.less(tf.abs(b), 1e-7), b, a/b)

def do_cov_inv(feats, mean, mask):
    temp = feats - mean
    sigma = tf.matmul(tf.expand_dims(temp, -1), tf.expand_dims(temp, -2))*tf.expand_dims(mask,-1)
    #sig_inv = tf.linalg.inv(sigma)*tf.expand_dims(mask,-1)
    return sigma#, sig_inv


def process_annot(annot_tensor, feat, mean, num_classes):
    one_hot = tf.one_hot(annot_tensor, num_classes)
    resized = tf.expand_dims(tf.image.resize_nearest_neighbor(one_hot, feat.get_shape().as_list()[1:-1]), -1)
    sorted_feats = tf.expand_dims(feat, -2)*resized
    sums = tf.reduce_sum(resized, [1,2])
    mean_feats = safe_div(tf.reduce_sum(sorted_feats*resized, [1,2]), sums)
    mean_feats = tf.expand_dims(tf.expand_dims(mean_feats, 1), 1)
    sorted_mask = tf.cast(tf.not_equal(resized, 0), tf.float32)
    mean_mask = tf.cast(tf.not_equal(sums, 0), tf.float32)

    # sorted_sig = do_cov_inv(sorted_feats, mean, sorted_mask)
    # mean_sig = do_cov_inv(mean_feats, mean, mean_mask)

    return mean_feats, mean_mask, sorted_feats, sorted_mask

def img_to_patch(input_shape, patch_size):
    patch_place = tf.placeholder(tf.uint8, input_shape, "patch_in")
    expand = tf.expand_dims(patch_place, 0)
    patches = sliding_window.extract_patches(expand, patch_size[0], patch_size[1])
    return patches, patch_place

def tensor_to_patch(inputs, patch_size):
    #import pdb; pdb.set_trace()
    input_shape = inputs.get_shape().as_list()
    reshape_inputs = tf.reshape(inputs, [-1] + input_shape[1:3] + [np.prod(input_shape[3:])])
    patches = sliding_window.extract_patches(reshape_inputs, patch_size[0], patch_size[1])
    patches = tf.reshape(patches, [-1] + patch_size + input_shape[3:])
    return patches

def run_inference_graph(model, trained_checkpoint_prefix,
                        input_dict, num_images, input_shape, pad_to_shape,
                        label_color_map, output_directory, num_classes, patch_size):
    assert len(input_shape) == 3, "input shape must be rank 3"
    effective_shape = [None] + input_shape

    if isinstance(model._feature_extractor, resnet_ex_class):
        batch = 1
    elif isinstance(model._feature_extractor, mobilenet_ex_class):
        batch = 1

    use_pool = FLAGS.use_pool

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
        #m_k = torch.tensor(np.load(mean_file)["arr_0"])
        class_m_k = np.load(class_mean_file)["arr_0"]
        first_pass = False
        mean_p = tf.placeholder(tf.float32, class_m_k.shape, "mean_placeholder")
    else:
        mean_p = tf.placeholder(tf.float32, tf.shape(outputs[model.final_logits_key]), "mean_placeholder")

    mean_feats, mean_mask, sorted_feats, sorted_mask = process_annot(annot_tensor, outputs[model.final_logits_key], mean_p, num_classes)
    if patch_size:
        patches = tensor_to_patch(sorted_feats, patch_size)
        sorted_mask = tensor_to_patch(sorted_mask, patch_size)
        fetch = [patches, sorted_mask]
    else:
        fetch = [sorted_feats, sorted_mask]

    if use_pool:
        fetch = [mean_feats, mean_mask]

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    full_eye = None
    coord = tf.train.Coordinator()

    with tf.Session(config=config) as sess:
        saver = tf.train.Saver(tf.global_variables())

        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        tf.train.start_queue_runners(sess,coord=coord)
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

        #TODO: fix passes bug
        #for first_pass in passes:
        if first_pass:
            print("first pass")
        else:
            print("second pass")
            fetch = [sorted_feats, sorted_mask] #always get these two on second pass
        for idx in range(num_step):

            start_time = timeit.default_timer()

            #cache results for second pass
            if first_pass or cache is None:
                res = sess.run(fetch, {mean_p: class_m_k})
                if cache is not None:
                    cache.append(res)
            else:
                res = cache[idx]

            logits = res[0]
            mask = res[1]
            sig = res[2]
            for b in range(logits.shape[0]): #should only be 1
                ret = compute_stats(class_m_k, class_v_k_inv,
                                    logits[b:b+1], mask[b:b+1], sig[b:b+1], class_k, first_pass)

                class_m_k, class_v_k_inv, class_k = ret
            
            if full_eye is None:
                full_eye = tf.eye(num_classes, batch_shape=class_v_k_inv.shape[:-2])

            # if idx > 10:
            #     import pdb; pdb.set_trace()
            # if idx*batch > 5:
            #     break

            elapsed = timeit.default_timer() - start_time
            print('{}) wall time: {}'.format(elapsed/batch, (idx+1)*batch))
        coord.request_stop()
        if not first_pass:
            to_inv_pl = tf.placeholder(tf.float32, class_v_k_inv.shape)
            do_inv = tf.linalg.inv(to_inv_pl)
            try:
                class_v_k_inv = sess.run(do_inv, {to_inv_pl: class_v_k_inv.cpu().numpy()})
            except Exception as ex:
                print(ex)
                import pdb; pdb.set_trace()
                print("here")


    os.makedirs(stats_dir, exist_ok=True)

    if first_pass:
        class_m_k_np = class_m_k.numpy()
        if np.isnan(class_m_k_np).any():
            print("nan time")
            import pdb; pdb.set_trace()
        np.savez(class_mean_file, class_m_k_np)
    else:
        class_v_k_inv_np = class_v_k_inv
        import pdb; pdb.set_trace()
        if np.isnan(class_v_k_inv_np).any():
            print("nan time")
            import pdb; pdb.set_trace()

        np.savez(class_cov_file, class_v_k_inv_np)
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
