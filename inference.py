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

from protos import pipeline_pb2
from builders import model_builder
from libs.exporter import deploy_segmentation_inference_graph
from libs.constants import CITYSCAPES_LABEL_COLORS, CITYSCAPES_LABEL_IDS

slim = tf.contrib.slim

flags = tf.app.flags

FLAGS = flags.FLAGS

tf.flags.DEFINE_string('cityscapes_dir', '',
                       'Pattern matching ground truth images for Cityscapes.')

tf.flags.DEFINE_string('input_pattern', '',
                       'Cityscapes dataset root folder.')

tf.flags.DEFINE_string('annot_pattern', '',
                       'Pattern matching input images for Cityscapes.')

flags.DEFINE_string('input_shape', '1024,2048,3', # default Cityscapes values
                    'The shape to use for inference. This should '
                    'be in the form [height, width, channels]. A batch '
                    'dimension is not supported for this test script.')

flags.DEFINE_string('pad_to_shape', '1025,2049', # default Cityscapes values
                     'Pad the input image to the specified shape. Must have '
                     'the shape specified as [height, width].')

tf.flags.DEFINE_string('split_type', '',
                       'Type of split: `train`, `test` or `val`.')

flags.DEFINE_string('config_path', None,
                    'Path to a pipeline_pb2.TrainEvalPipelineConfig config '
                    'file.')

flags.DEFINE_string('trained_checkpoint', None,
                    'Path to trained checkpoint, typically of the form '
                    'path/to/model.ckpt')

flags.DEFINE_string('output_dir', None, 'Path to write outputs images.')

flags.DEFINE_boolean('label_ids', False,
                     'Whether the output should be label ids.')

flags.DEFINE_boolean('compute_stats', False,
                     'Compute stats for mahalanobis distance')

_DEFAULT_PATTEN = {
    'input': '*_leftImg8bit.png',
    'label': '*_gtFine_labelTrainIds.png',
}

_DEFAULT_DIR = {
    'input': 'leftImg8bit',
    'label': 'gtFine',
}

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

# def compute_stats(final_logits):
#     #m_k = m_k-1 + (x_k - m_k-1)/k
#     #v_k = v_k-1 + (x_k - m_k-1)*(x_k - m_k)
#     shape = final_logits.get_shape().as_list()
#     if shape[0] != 1:
#         raise RuntimeError("Must have batch size of 1")
    
#     with tf.variable_scope("ComputeStats"):
#         k = tf.get_variable("k", initializer=1.)
#         mk = tf.get_variable("mk", shape[1:])
#         mkm1 = tf.get_variable("mkm1", shape[1:])
#         vk = tf.get_variable("vk", shape[1:])
#         xk = final_logits[0]
#         update_mkm1 = tf.assign(mkm1, mk)
#         temp = (xk - mkm1)
#         update_mk = tf.assign_add(mk, temp/k)
#         update_vk = tf.assign_add(vk, temp*(xk - mk))
#     return mk, vk, [update_mkm1, update_mk, update_vk]
def b_inv(b_mat):
    eye = b_mat.new_ones(b_mat.size(-1)).diag().expand_as(b_mat)
    b_inv, _ = torch.gesv(eye, b_mat)
    return b_inv

def compute_mean(m_km1, x_k, k):
    if m_km1 is None:
        return torch.tensor(x_k)
    x_k = torch.tensor(x_k)
    temp = (x_k - m_km1)
    m_k = m_km1 + temp/k
    return m_k

def compute_cov_inv(m, v_km1_inv, x_k, k):
    dim = x_k.shape[-1]
    out_shape = list(x_k.shape[:-1]) + [dim, x_k.shape[-1]]
    if v_km1_inv is None:
        return torch.zeros(out_shape)
    x_k = torch.tensor(x_k)

    temp = (x_k - m)
    sigma = torch.bmm(temp.view(-1, dim).unsqueeze(2), temp.view(-1, dim).unsqueeze(1)).view(out_shape)
    #import pdb; pdb.set_trace()
    v_k_inv = b_inv(sigma) + v_km1_inv
    return v_k_inv

def compute_stats(m_km1, v_km1_inv, x_k, k, first_pass):
    #numpy version might be better as tf order of updates matters
    #not sure how to control that
    # dim = x_k.shape[-1]
    # out_shape = list(x_k.shape) + [dim]
    # if m_km1 is None or v_km1 is None:
    #     return torch.tensor(x_k), torch.zeros(out_shape)
    # x_k = torch.tensor(x_k)
    
    # temp = (x_k - m_km1)
    # m_k = m_km1 + temp/k
    # temp2 = torch.bmm(temp.view(-1, dim).unsqueeze(2), (x_k - m_k).view(-1, dim).unsqueeze(1)).view(out_shape)
    # v_k = v_km1 + temp2 #temp*(x_k - m_k)
    # return m_k, v_k
    if first_pass:
        m_k = compute_mean(m_km1, x_k, k)
        v_k_inv = v_km1_inv
    else:
        m_k = m_km1
        v_k_inv = compute_cov_inv(m_k, v_km1_inv, x_k, k)
    return m_k, v_k_inv

def process_annot(pred_shape, feat, num_classes):
    annot_place = tf.placeholder(tf.uint8, pred_shape[:-1], "annot_in")
    one_hot = tf.one_hot(annot_place, num_classes)
    resized = tf.expand_dims(tf.image.resize_bilinear(one_hot, feat.get_shape().as_list()[1:-1]), -1)
    sorted_feats = tf.expand_dims(feat, -2)*resized
    return annot_place, sorted_feats

def run_inference_graph(model, trained_checkpoint_prefix,
                        input_images, annot_filenames, input_shape, pad_to_shape,
                        label_color_map, output_directory, num_classes):

    outputs, placeholder_tensor = deploy_segmentation_inference_graph(
        model=model,
        input_shape=input_shape,
        pad_to_shape=pad_to_shape,
        label_color_map=label_color_map)
    pred_tensor = outputs[model.main_class_predictions_key]

    # pl_size = np.reduce_prod(placeholder_tensor.get_shape().as_list())
    # placeholder_tensor = tf.random_uniform(tf.shape(placeholder_tensor),maxval=pl_size)

    stats_dir = os.path.join(output_directory, "stats")
    mean_file = os.path.join(stats_dir, "mean.npz")
    class_mean_file = os.path.join(stats_dir, "class_mean.npz")
    cov_file = os.path.join(stats_dir, "cov_inv.npz")
    class_cov_file = os.path.join(stats_dir, "class_cov_inv.npz")

    x = None
    y = None
    m_k = None
    v_k_inv = None
    class_m_k = None
    class_v_k_inv = None
    first_pass = True
    if FLAGS.compute_stats:
        if os.path.exists(mean_file) and os.path.exists(class_mean_file):
            m_k = torch.tensor(np.load(mean_file)["arr_0"])
            class_m_k = torch.tensor(np.load(class_mean_file)["arr_0"])
            first_pass = False
            print("second pass")
        else:
            print("first pass")
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    with tf.Session(config=config) as sess:
        input_graph_def = tf.get_default_graph().as_graph_def()
        saver = tf.train.Saver(tf.global_variables())
        
        fetch = [pred_tensor]
        if FLAGS.compute_stats:
            # feats = outputs[model.final_logits_key]
            # shape = feats.get_shape().as_list()
            # feats = tf.reshape(feats,[-1, shape[-1]])
            # temp = tf.constant(0, tf.float32, feats.get_shape().as_list())
            # covar, update = tf.contrib.metrics.streaming_covariance(feats, temp)
            # mean = [v for v in tf.get_collection(tf.GraphKeys.METRIC_VARIABLES) if "mean_prediction" in v.op.name][0]
            # fetch += [update]
            # sess.run(tf.global_variables_initializer())
            # sess.run(tf.local_variables_initializer())
            pred_shape = pred_tensor.get_shape().as_list()
            annot_place, sorted_feats = process_annot(pred_shape, outputs[model.final_logits_key], num_classes)
            fetch += [outputs[model.final_logits_key], sorted_feats]
        saver.restore(sess, trained_checkpoint_prefix)

        for idx in range(len(input_images)):
            image_path = input_images[idx]
            image_raw = np.array(Image.open(image_path))
            annot_raw = cv2.imread(annot_filenames[idx])

            start_time = timeit.default_timer()
            feed = {placeholder_tensor: image_raw}
            if FLAGS.compute_stats:
                feed[annot_place] = np.expand_dims(annot_raw[:,:,0], 0)
            res = sess.run(fetch,
                feed_dict=feed)
            predictions = res[0]
            if FLAGS.compute_stats:
                logits = res[1]
                sorted_logtis = res[2]
                m_k, v_k_inv = compute_stats(m_k, v_k_inv, logits, idx+1, first_pass)
                class_m_k, class_v_k_inv = compute_stats(class_m_k, class_v_k_inv, sorted_logtis, idx+1, first_pass)
            
            # if idx > 10:
            #     import pdb; pdb.set_trace()
            elapsed = timeit.default_timer() - start_time
            print('{}) wall time: {}'.format(elapsed, idx+1))
            if not FLAGS.compute_stats:
                filename = os.path.basename(image_path)
                save_location = os.path.join(output_directory, filename)

                predictions = predictions.astype(np.uint8)
                output_channels = len(label_color_map[0])
                if output_channels == 1:
                    predictions = np.squeeze(predictions[0],-1)
                else:
                    predictions = predictions[0]
                im = Image.fromarray(predictions)
                im.save(save_location, "PNG")
            
            if x is None or y is None:
                x = np.arange(0, predictions.shape[2], dtype=np.int32)
                y = np.arange(0, predictions.shape[1], dtype=np.int32)
                x, y = np.meshgrid(x,y)
            
            # if idx > 10:
            #     break

        # m_k, v_k_inv = sess.run([mean, covar])

        if FLAGS.compute_stats:
            os.makedirs(stats_dir, exist_ok=True)

            if first_pass:
                class_m_k = class_m_k.numpy()
                m_k = m_k.numpy()
                if np.isnan(m_k).any() or np.isnan(class_m_k).any():
                    print("nan time")
                    import pdb; pdb.set_trace()
                np.savez(mean_file, m_k)
                np.savez(class_mean_file, class_m_k)
            else:
                #v_k = b_inv(v_k_inv)
                #class_v_k = b_inv(class_v_k_inv)

                class_v_k_inv = class_v_k_inv.numpy()
                v_k_inv = v_k_inv.numpy()

                if np.isnan(v_k_inv).any() or np.isnan(class_v_k_inv).any():
                    print("nan time")
                    import pdb; pdb.set_trace()

                np.savez(class_cov_file, class_v_k_inv/(idx+1))
                np.savez(cov_file, v_k_inv/(idx+1))
        # print(save_location)
        # res = 25
        # vec_pred = np.fliplr(vec_pred)
        # plt.quiver(vec_pred[0,::res,::res,0], vec_pred[0,::res,::res,1])
        # plt.show()


def main(_):
    assert FLAGS.output_dir, '`output_dir` missing.'
    assert FLAGS.split_type, '`split_type` missing.'
    assert (FLAGS.cityscapes_dir) or \
           (FLAGS.input_pattern and FLAGS.annot_pattern), \
           'Must specify either `cityscapes_dir` or ' \
           '`input_pattern` and `annot_pattern`.'

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

    if FLAGS.pad_to_shape:
        pad_to_shape = [
            int(dim) if dim != '-1' else None
                for dim in FLAGS.pad_to_shape.split(',')]

    if FLAGS.cityscapes_dir:
        search_image_files = os.path.join(FLAGS.cityscapes_dir,
            _DEFAULT_DIR['input'], FLAGS.split_type, '*', _DEFAULT_PATTEN['input'])
        search_annot_files = os.path.join(FLAGS.cityscapes_dir,
            _DEFAULT_DIR['label'], FLAGS.split_type, '*', _DEFAULT_PATTEN['label'])
        input_images = glob.glob(search_image_files)
        annot_filenames = glob.glob(search_annot_files)
    else:
        input_images = glob.glob(FLAGS.input_pattern)
        annot_filenames = glob.glob(FLAGS.annot_pattern)
    
    if len(input_images) != len(annot_filenames):
        print("images: ", len(input_images))
        print("annot: ", len(annot_filenames))
        raise ValueError('Supplied patterns do not have image counts.')

    input_images = sorted(input_images)
    annot_filenames = sorted(annot_filenames)

    label_map = (CITYSCAPES_LABEL_IDS
        if FLAGS.label_ids else CITYSCAPES_LABEL_COLORS)

    num_classes, segmentation_model = model_builder.build(
        pipeline_config.model, is_training=False)

    run_inference_graph(segmentation_model, FLAGS.trained_checkpoint,
                        input_images, annot_filenames, input_shape, pad_to_shape,
                        label_map, output_directory, num_classes)

if __name__ == '__main__':
    tf.app.run()
