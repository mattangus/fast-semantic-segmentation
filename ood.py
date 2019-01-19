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

flags.DEFINE_string('eval_dir', None, 'Path to write outputs images.')

flags.DEFINE_boolean('label_ids', False,
                     'Whether the output should be label ids.')

flags.DEFINE_boolean('global_cov', False,'')

flags.DEFINE_boolean('global_mean', False,'')

flags.DEFINE_boolean('write_out', False,'')

flags.DEFINE_boolean('debug', False,'')

flags.DEFINE_boolean('use_patch', False,
                     'avg pool over spatial dims')

flags.DEFINE_boolean('do_ood', False,
                     'use ood dataset if true, otherwise use eval set')

flags.DEFINE_boolean('train_kernel', False,
                     'train a kernel for extracting edges')

# GRADIENT_CHECKPOINTS = [
#     "SharedFeatureExtractor/MobilenetV2/expanded_conv_4/output",
#     "SharedFeatureExtractor/MobilenetV2/expanded_conv_8/output",
#     "SharedFeatureExtractor/MobilenetV2/expanded_conv_12/output",
#     "SharedFeatureExtractor/MobilenetV2/expanded_conv_16/output",
# ]

def linkern(kernlen, space, power=2):
    x = np.linspace(-((kernlen-1)/2 + space), (kernlen-1)/2 + space, kernlen)
    grid = np.meshgrid(x,x)
    kernel = -np.power(np.power(np.abs(grid[0]),power) + np.power(np.abs(grid[1]), power), 1/power)
    kernel = kernel - np.min(kernel)
    kernel = kernel/np.max(kernel)
    return kernel

def gaus(x,x0,sigma):
    return np.exp(-(x-x0)**2/(2*sigma**2))

def gkern(kernlen=27, sigma=1.9491):
    """Returns a 2D Gaussian kernel array."""
    x = np.linspace(-((kernlen-1)/2), (kernlen-1)/2, kernlen)
    grid = np.meshgrid(x,x)
    kernel = np.abs(grid[0]) + np.abs(grid[1])
    gkernel = gaus(kernel, 0, sigma)
    return gkernel

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
#     min_dist_v = tf.reduce_min(full_dist, -1)
#     # scaled_dist = full_dist/tf.reduce_max(full_dist)
#     # dist_out = (scaled_dist*255).astype(np.uint8)
#     return dist_class, full_dist, min_dist_v, mean_p, var_inv_p #, [temp, temp2, left, dist, img_dist]

def process_logits(final_logits, mean_v, var_inv_v, depth, pred_shape, num_classes, global_cov, global_mean):
    mean_p = tf.placeholder(tf.float32, mean_v.shape, "mean")
    var_inv_p = tf.placeholder(tf.float32, var_inv_v.shape, "var_inv")
    var_inv = var_inv_p
    mean = mean_p

    if FLAGS.use_patch:
        orig_shape = final_logits.get_shape().as_list()
        new_shape = orig_shape
        new_shape[-1] *= np.prod(stats.PATCH_SIZE)
        final_logits = stats.get_patches(final_logits, stats.PATCH_SIZE)
        final_logits = tf.reshape(final_logits, new_shape)


    in_shape = final_logits.get_shape().as_list()
    #var_inv = tf.reshape(var_inv, [-1, in_shape[-1], in_shape[-1]])
    #mean = tf.reshape(mean, [-1, num_classes, in_shape[-1]])

    mean_sub = tf.expand_dims(final_logits,-2) - mean
    #final_logits = tf.reshape(final_logits, [-1, depth])
    #mean_sub = tf.expand_dims(tf.reshape(mean_sub, [-1, in_shape[-1]]), 1)
    mean_sub = tf.expand_dims(mean_sub, -2)

    tile_size = [in_shape[0]] + ([1] * (mean_sub._rank()-1))
    var_inv = tf.tile(var_inv, tile_size)
    left = tf.matmul(mean_sub, var_inv)
    mahal_dist = tf.squeeze(tf.sqrt(tf.matmul(left, mean_sub, transpose_b=True)))

    # const = np.log(np.power((2 * np.pi), -var_inv_v.shape[-1]))
    # power = (const + tf.linalg.logdet(var_inv) - tf.sqrt(mahal_dist))/2
    # dist = tf.exp(tf.cast(power, tf.float64))
    dist = mahal_dist

    img_dist = tf.expand_dims(tf.reshape(dist, in_shape[1:-1] + [num_classes]), 0)
    bad_pixel = tf.logical_or(tf.equal(img_dist, tf.zeros_like(img_dist)), tf.is_nan(img_dist))
    img_dist = tf.where(bad_pixel, tf.ones_like(img_dist)*float("inf"), img_dist)
    full_dist = tf.image.resize_bilinear(img_dist, (pred_shape[1],pred_shape[2]))
    dist_class = tf.expand_dims(tf.argmin(full_dist, -1),-1)
    min_dist = tf.reduce_min(full_dist, -1)
    # scaled_dist = full_dist/tf.reduce_max(full_dist)
    # dist_out = (scaled_dist*255).astype(np.uint8)
    #import pdb; pdb.set_trace()
    #return dist_class, img_dist, min_dist, mean_p, var_inv_p, []#[mahal_dist, power] #, [temp, temp2, left, dist, img_dist]
    return dist_class, full_dist, min_dist, mean_p, var_inv_p, []#[mahal_dist, power] #, [temp, temp2, left, dist, img_dist]

def pred_to_ood(pred, thresh=None):
    median = tf.contrib.distributions.percentile(pred, 50.0, interpolation='lower')
    median += tf.contrib.distributions.percentile(pred, 50.0, interpolation='higher')
    median /= 2.

    if thresh is None:
        return tf.to_float(pred >= median)
    else:
        return tf.to_float(pred >= thresh)

def get_miou(labels,
             predictions,
             num_classes,
             ignore_label,
             do_ood):
    ne = [tf.not_equal(labels, il) for il in ignore_label]
    neg_validity_mask = ne.pop(0)
    for v in ne:
        neg_validity_mask = tf.logical_and(neg_validity_mask, v)

    if do_ood:
        #0 = in distribution, 1 = OOD
        labels = tf.to_float(labels >= num_classes)
        num_classes = 2

    eval_labels = tf.where(neg_validity_mask, labels,
                            tf.zeros_like(labels))

    #eval_labels = tf.Print(eval_labels, ["unique", tf.unique(tf.reshape(eval_labels, [-1]))[0]], summarize=21)

    return mean_iou(eval_labels, predictions, num_classes, weights=tf.to_float(neg_validity_mask))

def get_colours(n):
    rgbs = []
    for i in range(0, 360, 360 // n):
        h = i/360
        s = (40 + random.random() * 10)/100
        l = (40 + random.random() * 10)/100
        rgbs.append(colorsys.hls_to_rgb(h, l, s))
    random.shuffle(rgbs)
    return rgbs

def display_projection(final_out, annot_out, components, ignore=255, tformed=None):
    assert components in [2,3], "only allow 2d or 3d reduction"
    if tformed is None:
        tformed = TSNE(n_components=components, n_jobs=32, verbose=2, perplexity=100.0, n_iter=5000).fit_transform(np.reshape(final_out,[-1,32]))
    colour = np.reshape(cv2.resize(annot_out, (final_out.shape[1], final_out.shape[0]), interpolation=cv2.INTER_NEAREST),[-1])
    tformed = tformed[colour != ignore]
    colour = colour[colour != ignore]
    colour_map = get_colours(int(np.max(colour[colour != ignore])))
    # colour = np.array([colour_map[int(c)] for c in colour])
    #idx = np.random.choice(np.arange(tformed.shape[0]), 1000, replace=False)
    fig = plt.figure()
    if components == 3:
        ax = fig.add_subplot(111, projection="3d")
    elif components == 2:
        ax = fig.add_subplot(111)
    selected_tformed = tformed
    selected_colour = colour
    for g in np.unique(selected_colour):
        idx = np.where(selected_colour == g)
        if g == 3:
            marker = "x"
            s = 50
        else:
            marker = None
            s = 5
        im = ax.scatter(selected_tformed[idx,0], selected_tformed[idx,1], c=colour_map[g], label=train_name[g], marker=marker, s=s)
    ax.legend()
    plt.show()
    return tformed, colour

def write_hist(x, title, path, div=2000):
    x = np.array(x)
    plt.hist(np.reshape(x,[-1]), np.prod(x.shape)//div)
    plt.title(title)
    plt.xlabel("value")
    plt.ylabel("Frequency")
    plt.savefig(path)
    plt.cla()
    plt.clf()

def to_img(x):
    if len(x.shape) == 2:
        x = np.expand_dims(x, -1)
    return (x/np.max(x)*255).astype(np.uint8)

def kernel_model(shape):
    with tf.variable_scope("kmodel"):
        ed = np.expand_dims
        img_pl = tf.placeholder(tf.float32, shape)
        edges_pl = tf.placeholder(tf.float32, shape)

        init = ed(ed(gkern(11),-1),-1)

        filter = tf.get_variable("kfilter",initializer=init.astype(np.float32))
        b = tf.get_variable("kbias", (), dtype=tf.float32)
        y = tf.nn.relu(tf.nn.conv2d(edges_pl,filter,[1,1,1,1],"SAME") + b)

        loss = tf.reduce_mean(tf.square(y - img_pl)) + 0.01*tf.linalg.norm(filter)

        train_step = tf.train.AdamOptimizer(0.01).minimize(loss)

        return img_pl, edges_pl, loss, train_step, filter

def train_kernel(img, edges, sess, img_pl, edges_pl, loss, train_step, filter):
    ed = np.expand_dims
    #all_losses = []
    img = ed(ed(img,0),-1)
    edges = ed(ed(edges,0),-1)

    img = img/np.max(img)
    edges = edges/np.max(edges)

    for i in range(10):
        cur_loss, _ = sess.run([loss, train_step], {img_pl: img, edges_pl: edges})
        print(i,":", cur_loss, end="\r")
        #all_losses.append(cur_loss)
    
    return sess.run(filter)

def run_inference_graph(model, trained_checkpoint_prefix,
                        input_dict, num_images, ignore_label, input_shape, pad_to_shape,
                        label_color_map, output_directory, num_classes, eval_dir,
                        min_dir, dist_dir, hist_dir, dump_dir):
    assert len(input_shape) == 3, "input shape must be rank 3"
    batch = 1
    do_ood = FLAGS.do_ood
    #from normalise_data.py
    mean_value = 646604.7
    std_value = np.sqrt(28830410695.369247)
    thresh = (-0.6853718 * std_value) + mean_value
    effective_shape = [batch] + input_shape

    input_queue = create_input(input_dict, batch, 15, 15, 15)
    input_dict = input_queue.dequeue()

    input_tensor = input_dict[dataset_builder._IMAGE_FIELD]
    annot_tensor = input_dict[dataset_builder._LABEL_FIELD]
    input_name = input_dict[dataset_builder._IMAGE_NAME_FIELD]

    annot_pl = tf.placeholder(tf.float32, annot_tensor.get_shape().as_list())
    outputs, placeholder_tensor = deploy_segmentation_inference_graph(
        model=model,
        input_shape=effective_shape,
        #input=input_tensor,
        pad_to_shape=pad_to_shape,
        input_type=tf.float32)

    pred_tensor = outputs[model.main_class_predictions_key]
    final_logits = outputs[model.final_logits_key]
    unscaled_logits = outputs[model.unscaled_logits_key]

    if FLAGS.use_patch:
        stats_dir = os.path.join(eval_dir, "stats.patch")
    else:
        stats_dir = os.path.join(eval_dir, "stats")
    class_mean_file = os.path.join(stats_dir, "class_mean.npz")
    class_cov_file = os.path.join(stats_dir, "class_cov_inv.npz")

    global_cov = FLAGS.global_cov
    global_mean = FLAGS.global_mean

    print("loading means and covs")
    mean = np.load(class_mean_file)["arr_0"]
    var_inv = np.load(class_cov_file)["arr_0"]
    print("done loading")
    var_dims = list(var_inv.shape[-2:])
    mean_dims = list(mean.shape[-2:])
    depth = mean_dims[-1]
    
    if global_cov:
        var_brod = np.ones_like(var_inv)
        var_inv = np.sum(var_inv, axis=(0,1,2), keepdims=True)*var_brod
    if global_mean:
        mean_brod = np.ones_like(mean)
        mean = np.mean(mean, axis=(0,1,2), keepdims=True)*mean_brod
        # import pdb; pdb.set_trace()

    #mean = np.reshape(mean, [-1] + mean_dims)
    #var_inv = np.reshape(var_inv, [-1] + var_dims)
    with tf.device("gpu:1"):
        dist_class, full_dist, min_dist, mean_p, var_inv_p, dbg  = process_logits(final_logits, mean, var_inv, depth, pred_tensor.get_shape().as_list(), num_classes, global_cov, global_mean)
        dist_colour = _map_to_colored_labels(dist_class, pred_tensor.get_shape().as_list(), label_color_map)
        pred_colour = _map_to_colored_labels(pred_tensor, pred_tensor.get_shape().as_list(), label_color_map)

    if do_ood:
        dist_class = tf.expand_dims(pred_to_ood(min_dist, thresh),-1)
        #pred is the baseline of assuming all ood
        pred_tensor = tf.ones_like(pred_tensor)

    with tf.variable_scope("PredIou"):
        pred_miou, pred_conf_mat, pred_update = get_miou(annot_pl, pred_tensor, num_classes, ignore_label, do_ood)
    with tf.variable_scope("DistIou"):
        dist_miou, dist_conf_mat, dist_update = get_miou(annot_pl, dist_class, num_classes, ignore_label, do_ood)

    iou_update = tf.group([pred_update, dist_update])

    mean = np.reshape(mean, mean_p.get_shape().as_list())
    var_inv = np.reshape(var_inv, var_inv_p.get_shape().as_list())

    input_fetch = [input_name, input_tensor, annot_tensor]

    fetch = [pred_tensor, pred_colour, dist_colour, dist_class, full_dist, min_dist, iou_update, final_logits, unscaled_logits]

    # Add checkpointing nodes to correct collection
    # if GRADIENT_CHECKPOINTS is not None:
    #     tf.logging.info(
    #         'Adding gradient checkpoints to `checkpoints` collection')
    #     graph = tf.get_default_graph()
    #     checkpoint_list = GRADIENT_CHECKPOINTS
    #     for checkpoint_node_name in checkpoint_list:
    #         curr_tensor_name = checkpoint_node_name + ":0"
    #         node = graph.get_tensor_by_name(curr_tensor_name)
    #         tf.add_to_collection('checkpoints', node)
    
    grads = tf.gradients(min_dist, placeholder_tensor)
    epsilon = 0.0
    if epsilon > 0.0:
        adv_img = placeholder_tensor - epsilon*tf.sign(grads)
    else:
        adv_img = tf.expand_dims(placeholder_tensor, 0)

    num_step = num_images // batch
    dump_dir += "_" + str(epsilon)
    tf.gfile.MakeDirs(dump_dir)

    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        tf.train.start_queue_runners(sess)
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(sess, trained_checkpoint_prefix)

        if FLAGS.train_kernel:
            kimg_pl, kedges_pl, kloss, ktrain_step, kfilter = kernel_model((1, 1024, 2048, 1))
            init = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="kmodel")
            sess.run(tf.variables_initializer(init))
            #sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        for idx in range(num_step):

            start_time = timeit.default_timer()

            inputs = sess.run(input_fetch)

            annot_raw = inputs[2]
            img_raw = inputs[1]
            image_path = inputs[0][0].decode("utf-8")
            filename = os.path.basename(image_path)
            dump_filename = os.path.join(dump_dir, filename)
            if os.path.exists(dump_filename):
                print("skipping", dump_filename)
                continue

            adv_img_out = sess.run(adv_img, feed_dict={mean_p: mean, var_inv_p: var_inv,
                            placeholder_tensor: img_raw, annot_pl: annot_raw})
            adv_img_out = adv_img_out[0]

            res, dbg_v = sess.run([fetch, dbg], feed_dict={mean_p: mean, var_inv_p: var_inv,
                            placeholder_tensor: adv_img_out, annot_pl: annot_raw})

            pred_miou_v, dist_miou_v = sess.run([pred_miou, dist_miou])
            #import pdb; pdb.set_trace()
            elapsed = timeit.default_timer() - start_time
            end = "\r"
            if idx % 50 == 0:
                #every now and then do regular print
                end = "\n"
            print('{0:.4f} iter: {1}, pred iou: {2:.6f}, dist iou: {3:.6f}'.format(elapsed, idx+1, pred_miou_v, dist_miou_v))

            if FLAGS.train_kernel:
                predictions = res[0]
                min_dist_out = res[5][0]
                edges = cv2.Canny(predictions[0].astype(np.uint8),1,1)
                #import pdb; pdb.set_trace()
                filter = train_kernel(min_dist_out, edges, sess, kimg_pl, kedges_pl, kloss, ktrain_step, kfilter)
                #all_filters.append(filter)
                # kernel = gkern(sigma=0.2)
                dilated = np.expand_dims(cv2.filter2D(edges,-1,filter[...,0,0]),-1).astype(np.float32)
                dilated = dilated/np.max(dilated)
                
                disp = cv2.resize(np.concatenate([to_img(min_dist_out), to_img(dilated)], 1), (int(1920), int(1080)))
                cv2.imshow("test", disp)
                cv2.waitKey(1)

            if FLAGS.write_out:
                predictions = res[0]
                prediction_colour = res[1]
                dist_out = res[2][0].astype(np.uint8)
                full_dist_out = res[4][0]
                min_dist_out = res[5][0]
                unscaled_logits_out = res[8][0]

                # annot_out = res[8][0]
                # n_values = np.max(annot_out) + 1
                # one_hot_out = np.eye(n_values)[annot_out][...,0,:num_classes]
                
                #import pdb; pdb.set_trace()

                min_dist_v = np.expand_dims(np.nanmin(full_dist_out, -1), -1)
                min_dist_v[np.logical_not(np.isfinite(min_dist_v))] = np.nanmin(full_dist_out)
                min_dist_v = min_dist_v - np.min(min_dist_v) #min now at 0
                min_dist_v = (255*min_dist_v/np.max(min_dist_v)).astype(np.uint8) #max now at 255
                
                save_location = os.path.join(output_directory, filename)
                dist_filename = os.path.join(dist_dir, filename)
                min_filename = os.path.join(min_dir, filename)

                #write_hist(min_dist_out, "Min Dist", os.path.join(hist_dir, filename))

                #all_mins.append(min_dist_out)

                # if idx == 30:
                #     write_hist(all_mins, "Combined Dists", os.path.join(hist_dir, "all"))

                prediction_colour = prediction_colour.astype(np.uint8)
                output_channels = len(label_color_map[0])
                if output_channels == 1:
                    prediction_colour = np.squeeze(prediction_colour[0],-1)
                else:
                    prediction_colour = prediction_colour[0]
                #import pdb; pdb.set_trace()
                cv2.imwrite(save_location, prediction_colour)
                cv2.imwrite(min_filename, min_dist_v)
                cv2.imwrite(dist_filename, dist_out)
                np.savez(dump_filename, {"full": full_dist_out, "unscaled_logits": unscaled_logits_out})
            
            if FLAGS.debug:
                dist_out = res[2][0].astype(np.uint8)
                full_dist_out = res[4][0]
                min_dist_out = res[5][0]

                min_dist_v = np.expand_dims(np.nanmin(full_dist_out, -1), -1)
                min_dist_v[np.logical_not(np.isfinite(min_dist_v))] = np.nanmin(full_dist_out)
                min_dist_v = min_dist_v - np.min(min_dist_v) #min now at 0
                min_dist_v = (255*min_dist_v/np.max(min_dist_v)).astype(np.uint8) #max now at 255
                
                final_out = res[7][0]
                annot_out = inputs[2][0]
                img_out = inputs[1][0]
                
                thresh = np.median(min_dist_out)
                grain = (np.max(min_dist_out) - np.min(min_dist_out))/300
                print(thresh, "  ", grain)
                while True:
                    mask = np.expand_dims(min_dist_out < thresh,-1)
                    #cv2.imshow("img", (255*mask).astype(np.uint8))
                    cv2.imshow("img", (img_out*mask).astype(np.uint8))
                    key = cv2.waitKey(1)
                    if key == 27: #escape
                        break
                    elif key == 115: #s
                        thresh += grain
                        print(thresh, "  ", grain)
                    elif key == 119: #w
                        thresh -= grain
                        print(thresh, "  ", grain)
                    elif key == 97: #a
                        grain -= 5
                        print(thresh, "  ", grain)
                    elif key == 100: #d
                        grain += 5
                        print(thresh, "  ", grain)
                    elif key == 112: #p
                        import pdb; pdb.set_trace()
        print('{0:.4f} iter: {1}, pred iou: {2:.6f}, dist iou: {3:.6f}'.format(elapsed, idx+1, pred_miou_v, dist_miou_v))


def main(_):
    eval_dir = FLAGS.eval_dir
    output_directory = os.path.join(eval_dir, "inf")
    suff = ""
    if FLAGS.global_mean:
        suff = "_G"
    else:
        suff = "_L"
    if FLAGS.global_cov:
        suff += "G"
    else:
        suff += "L"
    dist_dir = os.path.join(eval_dir, "class_dist" + suff)
    min_dir = os.path.join(eval_dir, "min" + suff)
    hist_dir = os.path.join(eval_dir, "hist" + suff)
    dump_dir = os.path.join(eval_dir, "dump" + suff)

    tf.gfile.MakeDirs(output_directory)
    tf.gfile.MakeDirs(min_dir)
    tf.gfile.MakeDirs(dist_dir)
    tf.gfile.MakeDirs(hist_dir)
    #tf.gfile.MakeDirs(dump_dir)
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
        if FLAGS.label_ids else CITYSCAPES_LABEL_COLORS)

    num_classes, segmentation_model = model_builder.build(
        pipeline_config.model, is_training=False)

    if FLAGS.do_ood:
        input_reader = pipeline_config.ood_eval_input_reader
    else:
        input_reader = pipeline_config.eval_input_reader
        
    input_reader.shuffle = True
    input_reader.num_epochs = 1
    input_dict = dataset_builder.build(input_reader)

    ignore_label = pipeline_config.ood_config.ignore_label

    run_inference_graph(segmentation_model, FLAGS.trained_checkpoint,
                        input_dict, input_reader.num_examples, ignore_label, input_shape, pad_to_shape,
                        label_map, output_directory, num_classes, eval_dir, min_dir, dist_dir, hist_dir,
                        dump_dir)

if __name__ == '__main__':
    tf.app.run()
