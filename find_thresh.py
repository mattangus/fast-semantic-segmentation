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
from sklearn.decomposition import PCA
#from sklearn.manifold import TSNE
from MulticoreTSNE import MulticoreTSNE as TSNE
from mpl_toolkits.mplot3d import Axes3D
import random
import colorsys

from protos import pipeline_pb2
from builders import model_builder, dataset_builder
from libs.exporter import deploy_segmentation_inference_graph, _map_to_colored_labels
from libs.constants import CITYSCAPES_LABEL_COLORS, CITYSCAPES_LABEL_IDS, URSA_LABEL_COLORS
from libs.metrics import mean_iou
from libs.ursa_map import train_name
from libs import stat_computer as stats

import scipy.stats as st

#from third_party.mem_gradients_patch import gradients

slim = tf.contrib.slim

flags = tf.app.flags

prefetch_queue = slim.prefetch_queue

flags.DEFINE_string('config_path', None,
                    'Path to a pipeline_pb2.TrainEvalPipelineConfig config '
                    'file.')

flags.DEFINE_string('eval_dir', None, 'Path to write outputs images.')

flags.DEFINE_string('thresh_dir', None, 'Path to write outputs images.')

flags.DEFINE_bool("use_mean", False, "use thresh*mean in model")

FLAGS = flags.FLAGS

epoch = 3

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

def pred_to_ood(pred, thresh=None):
    median = tf.contrib.distributions.percentile(pred, 50.0, interpolation='lower')
    median += tf.contrib.distributions.percentile(pred, 50.0, interpolation='higher')
    median /= 2.

    return tf.to_float(pred >= median)

def get_neg_valid(labels, ignore_label):
    ne = [tf.not_equal(labels, il) for il in ignore_label]
    neg_validity_mask = ne.pop(0)
    for v in ne:
        neg_validity_mask = tf.logical_and(neg_validity_mask, v)
    return neg_validity_mask

def get_miou(labels,
             predictions,
             num_classes,
             ignore_label,
             neg_validity_mask=None):

    if neg_validity_mask is None:
        neg_validity_mask = get_neg_valid(labels, ignore_label)

    #0 = in distribution, 1 = OOD
    labels = tf.to_float(labels >= num_classes)
    num_classes = 2
    
    #import pdb; pdb.set_trace()

    eval_labels = tf.where(neg_validity_mask, labels,
                            tf.zeros_like(labels))

    return mean_iou(eval_labels, predictions, num_classes, weights=tf.to_float(neg_validity_mask))

def build_model(annot_tensor, ignore_label, num_classes, mean_value, std_value, num_step):
    lamb = 0.2
    if FLAGS.use_mean:
        dims = 2
    else:
        dims = 1
    thresh = tf.get_variable("thresh", shape=(dims,), dtype=tf.float32)

    min_dist_pl = tf.placeholder(tf.float32, annot_tensor.shape.as_list())

    #normalise
    min_dist_norm = (min_dist_pl - mean_value) / std_value

    neg_validity_mask = get_neg_valid(annot_tensor, ignore_label)
    weights = tf.to_float(neg_validity_mask)

    #0 is ID, 1 is OOD
    norm_annot = tf.to_float(annot_tensor >= num_classes)

    not_batch = list(range(1,len(min_dist_norm.shape.as_list())))
    mean_dist = tf.reduce_mean(min_dist_norm, not_batch, keepdims=True)
    
    if FLAGS.use_mean:
        use_mean = thresh[0]*mean_dist
        thresh_value = thresh[1]
    else:
        use_mean = 0.0
        thresh_value = thresh

    print("use_mean:", use_mean)

    logits = (min_dist_norm - use_mean - thresh_value)
    predictions = tf.to_float(logits >= 0)

    pixel_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=norm_annot,
                                logits=logits, weights=weights)

    loss = tf.reduce_mean(pixel_loss)
    if FLAGS.use_mean:
        loss +=  0.1 * tf.norm(thresh)

    global_step = tf.train.get_or_create_global_step()
    lr = tf.train.exponential_decay(0.01,
        global_step, num_step, 0.01)

    train_step = tf.train.AdamOptimizer(lr).minimize(loss, global_step=global_step)

    correct = tf.to_float(tf.equal(predictions, norm_annot))
    acc = tf.reduce_sum(correct*weights)/tf.reduce_sum(weights)

    pct_out = tf.reduce_sum(norm_annot*weights)/tf.reduce_sum(weights)

    tf.summary.scalar("metrics/lr", lr)
    if FLAGS.use_mean:
        tf.summary.scalar("metrics/thresh_0", thresh[0])
        tf.summary.scalar("metrics/thresh_1", thresh[1])
    else:
        tf.summary.scalar("metrics/thresh", thresh[0])        
    tf.summary.scalar("metrics/loss", loss)
    tf.summary.scalar("metrics/acc", acc)
    tf.summary.scalar("metrics/pct_out", pct_out)
    tf.summary.scalar("metrics/gain", acc - pct_out)
    tf.summary.image("images/annot", annot_tensor)
    tf.summary.image("images/min_dist", min_dist_pl)
    tf.summary.image("images/predictions", predictions)
    
    return (thresh, loss, train_step, min_dist_pl, neg_validity_mask, acc, pct_out, lr, global_step,
            []#[norm_annot, logits, pixel_loss, min_dist_norm, predictions, correct]
            )
    
def get_dumps(image_paths, eval_dir):

    min_dist = []

    for path in image_paths:
        dump_file = os.path.join(eval_dir, os.path.basename(path) + ".npz")
        # if dump_file in dist_cache and False:
        #     min_dist.append(dist_cache[dump_file])
        # else:
        if not os.path.exists(dump_file):
            #print("skipping", dump_file)
            return None
        arr = np.expand_dims(np.load(dump_file)["arr_0"],-1)
        #dist_cache[dump_file] = arr 
        min_dist.append(arr)
    
    return np.array(min_dist)

def run_inference_graph(input_dict, num_images, ignore_label,
                        num_classes, eval_dir, thresh_dir):
    batch = 18
    do_ood = True
    mean_value = 646604.7
    std_value = np.sqrt(28830410695.369247)
    window = 30
    window = np.ones([window])/window
    check_file = os.path.join(thresh_dir, "model.ckpt")

    input_queue = create_input(input_dict, batch, 15, 15, 15)
    input_dict = input_queue.dequeue()

    input_tensor = input_dict[dataset_builder._IMAGE_FIELD]
    annot_tensor = input_dict[dataset_builder._LABEL_FIELD]
    input_name = input_dict[dataset_builder._IMAGE_NAME_FIELD]

    annot_pl = tf.placeholder(tf.float32, annot_tensor.get_shape().as_list())

    input_fetch = [input_name, input_tensor, annot_tensor]

    num_step = epoch * (num_images // batch)
    print("running for", num_step, "steps")

    temp = build_model(annot_pl, ignore_label, num_classes, mean_value, std_value, num_step)
    thresh, loss, train_step, min_dist_pl, neg_validity_mask, acc, pct_out, lr, global_step, dbg = temp

    saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint(thresh_dir)

    train_summary = tf.summary.FileWriter(thresh_dir)
    merged_summary = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        if ckpt is not None:
            saver.restore(sess, ckpt)
        tf.train.start_queue_runners(sess)

        start_step = sess.run(global_step)

        for idx in range(start_step, num_step):

            start_time = timeit.default_timer()

            inputs = sess.run(input_fetch)

            annot_raw = inputs[2]
            img_raw = inputs[1]
            image_paths = inputs[0]

            min_dist = get_dumps([p.decode("utf-8") for p in image_paths], eval_dir)
            
            if min_dist is None:
                #print("skipping", dump_file)
                continue

            fetch = [loss, train_step, thresh, acc, pct_out, lr, merged_summary, dbg]
            cur_loss, _, cur_thresh, cur_acc, num_out, cur_lr, cur_summary, dbg_out = sess.run(fetch, {annot_pl: annot_raw, min_dist_pl: min_dist})

            train_summary.add_summary(cur_summary, global_step=idx)

            if idx % 50 == 0:
                saver.save(sess, check_file, global_step=idx)
            #import pdb; pdb.set_trace()
            elapsed = timeit.default_timer() - start_time
            end = "\r"
            if idx % 50 == 0:
                #every now and then do regular print
                end = "\n"
            print('{0:.4f} iter: {1}, loss: {2:.4f}, acc: {3:.4f}, pct_out: {4:.4f} thresh: {5}, lr: {6}         '
                    .format(elapsed, idx+1, cur_loss, cur_acc, num_out, cur_thresh, cur_lr), end=end)


def main(_):
    eval_dir = FLAGS.eval_dir

    thresh_dir = FLAGS.thresh_dir
    tf.gfile.MakeDirs(thresh_dir)
    
    pipeline_config = pipeline_pb2.PipelineConfig()
    with tf.gfile.GFile(FLAGS.config_path, 'r') as f:
        text_format.Merge(f.read(), pipeline_config)

    num_classes = pipeline_config.model.pspnet.num_classes

    input_reader = pipeline_config.ood_train_input_reader
    input_reader.shuffle = True
    input_reader.num_epochs = epoch
    input_dict = dataset_builder.build(input_reader)

    ignore_label = pipeline_config.ood_config.ignore_label

    run_inference_graph(input_dict, input_reader.num_examples, ignore_label,
                        num_classes, eval_dir, thresh_dir)

if __name__ == '__main__':
    tf.app.run()
