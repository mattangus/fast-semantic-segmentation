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
from sklearn.manifold import TSNE
#from MulticoreTSNE import MulticoreTSNE as TSNE
from mpl_toolkits.mplot3d import Axes3D
import random
import colorsys
from multiprocessing import Process, Queue, Pool
import sklearn

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

flags.DEFINE_float("epsilon", 0.0, "")

flags.DEFINE_float("t_value", 1.0, "")

flags.DEFINE_boolean('global_cov', False,'')

flags.DEFINE_boolean('global_mean', False,'')

flags.DEFINE_boolean('write_out', False,'')

flags.DEFINE_boolean('write_img', False,'')

flags.DEFINE_boolean('max_softmax', False,'')

flags.DEFINE_boolean('odin', False,'')

flags.DEFINE_boolean('debug', False,'')

flags.DEFINE_boolean('use_patch', False,
                     'avg pool over spatial dims')

flags.DEFINE_boolean('do_ood', False,
                     'use ood dataset if true, otherwise use eval set')

flags.DEFINE_boolean('use_train', False,
                     'use ood dataset if true, otherwise use eval set')

flags.DEFINE_boolean('train_kernel', False,
                     'train a kernel for extracting edges')

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
    mean_p = tf.placeholder(tf.float32, mean_v.shape, "mean_p")
    var_inv_p = tf.placeholder(tf.float32, var_inv_v.shape, "var_inv_p")
    mean = tf.get_variable("mean", initializer=mean_p, trainable=False)
    var_inv = tf.get_variable("var_inv", initializer=var_inv_p, trainable=False)

    no_load = [mean, var_inv]

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
    return dist_class, img_dist, full_dist, min_dist, mean_p, var_inv_p, set(no_load), []#[mahal_dist, power] #, [temp, temp2, left, dist, img_dist]

def pred_to_ood(pred, mean_value, std_value, thresh=None):

    if thresh is None:
        median = tf.contrib.distributions.percentile(pred, 50.0, interpolation='lower')
        median += tf.contrib.distributions.percentile(pred, 50.0, interpolation='higher')
        median /= 2.
        return tf.to_float(pred >= median)
    else:
        min_dist_norm = (pred - mean_value) / std_value
        return tf.nn.sigmoid(min_dist_norm * 0.13273349404335022 + 0.38076120615005493)

def get_valid(labels, ignore_label):
    ne = [tf.not_equal(labels, il) for il in ignore_label]
    neg_validity_mask = ne.pop(0)
    for v in ne:
        neg_validity_mask = tf.logical_and(neg_validity_mask, v)
    
    return neg_validity_mask

def get_miou(labels,
             predictions,
             num_classes,
             ignore_label,
             do_ood,
             neg_validity_mask=None):
    if neg_validity_mask is None:
        neg_validity_mask = get_valid(labels, ignore_label)

    #if do_ood:
    #0 = in distribution, 1 = OOD
    labels = tf.to_float(labels >= num_classes)
    num_classes = 2

    eval_labels = tf.where(neg_validity_mask, labels,
                            tf.zeros_like(labels))

    #eval_labels = tf.Print(eval_labels, ["unique", tf.unique(tf.reshape(eval_labels, [-1]))[0]], summarize=21)

    weights = tf.to_float(neg_validity_mask)

    return mean_iou(eval_labels, predictions, num_classes, weights=weights), neg_validity_mask

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

class ParallelWriter(object):

    def __init__(self, write_queue):
        self.write_queue = write_queue
        self.p = Process(target=self.write_thread)
        self.p.start()

    def write_thread(self):
        while True:
            item = self.write_queue.get()
            if item is None:
                break
            idx, filename, data = item
            if filename.endswith(".png") or filename.endswith(".jpg"):
                #print(idx, filename, type(data))
                cv2.imwrite(filename, np.array(data))
            elif filename.endswith(".npz"):
                np.savez(filename, data)
            elif filename.endswith(".npy"):
                np.save(filename, data)
            else:
                print("unknown format", filename)
        print("thread done")

    def put(self, idx, filename, data):
        self.write_queue.put((idx,filename,data))

    def close(self):
        self.write_queue.put(None)
        self.p.join()
        print("closed writer")

    def size(self):
        return self.write_queue.qsize()

def make_plots(roc, pr, num_thresholds):
    eps = 1e-7
    #from http://www.medicalbiostatistics.com/roccurve.pdf page 6
    optimal = np.argmin(np.sqrt(np.square(1-roc[:,1]) + np.square(roc[:,0])))
    #from https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/metrics/python/ops/metric_ops.py
    threshs = [(i + 1) * 1.0 / (num_thresholds - 1) for i in range(num_thresholds - 2)]
    threshs = [0.0 - eps] + threshs + [1.0 + eps]
    print("optimal threshold:", threshs[optimal])
    AUC = -np.trapz(roc[:,1], roc[:,0])
    print("area under curve:", AUC)
    optimal_point = roc[optimal]
    min_v = -0.01
    max_v = 1.01

    #roc curve
    plt.subplot(1, 2, 1)
    plt.plot(roc[:,0], roc[:,1])
    plt.scatter(optimal_point[0], optimal_point[1], marker="o", c="r")
    plt.plot(threshs, threshs, linestyle="--", color="black", linewidth=1)
    plt.ylim((min_v, max_v))
    plt.xlim((min_v, max_v))
    plt.title("ROC")

    #pr curve
    plt.subplot(1,2,2)
    plt.plot(pr[:,0], pr[:,1])
    plt.ylim((min_v, max_v))
    plt.xlim((min_v, max_v))
    plt.title("PR")

    #plt.show()

    print("ROCpoints:", repr(roc))
    print("PRpoints:", repr(pr))

def run_inference_graph(model, trained_checkpoint_prefix,
                        input_dict, num_images, ignore_label, input_shape, pad_to_shape,
                        label_color_map, output_directory, num_classes, eval_dir,
                        min_dir, dist_dir, hist_dir, dump_dir):
    assert len(input_shape) == 3, "input shape must be rank 3"
    batch = 1
    do_ood = FLAGS.do_ood
    epsilon = FLAGS.epsilon
    #epsilon = np.linspace(0,0.0001,10)
    dump_dir += "_" + str(epsilon)
    #from normalise_data.py
    # norms = np.load(os.path.join(dump_dir, "normalisation.npy")).item()
    # mean_value = norms["mean"]
    # std_value = norms["std"]
    mean_value = 508.7571
    std_value = 77.60572284853058
    if FLAGS.max_softmax:
        thresh = 0.0650887573964497 #dim from sun train
    else:
        thresh = 0.37583892617449666 #dim from sun train
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

    stats_dir = os.path.join(eval_dir, "stats.dtform")
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
        not_correct = tf.to_float(tf.not_equal(annot_pl, tf.to_float(pred_tensor)))
        dist_class, img_dist, full_dist, min_dist, mean_p, var_inv_p, vars_noload, dbg  = process_logits(final_logits, mean, var_inv, depth, pred_tensor.get_shape().as_list(), num_classes, global_cov, global_mean)
        dist_colour = _map_to_colored_labels(dist_class, pred_tensor.get_shape().as_list(), label_color_map)
        pred_colour = _map_to_colored_labels(pred_tensor, pred_tensor.get_shape().as_list(), label_color_map)

        if FLAGS.max_softmax:
            interp_logits = tf.image.resize_bilinear(unscaled_logits, pred_tensor.shape.as_list()[1:3])
            dist_pred = 1 - tf.reduce_max(tf.nn.softmax(interp_logits/FLAGS.t_value),-1, keepdims=True)
            dist_class = tf.to_float(dist_pred >= thresh)
        else:
            dist_pred = tf.expand_dims(pred_to_ood(min_dist, mean_value, std_value, thresh),-1)
            dist_class = tf.to_float(dist_pred >= thresh)

        #pred is the baseline of assuming all ood
        pred_tensor = tf.ones_like(pred_tensor)

    with tf.device("gpu:1"):
        neg_validity_mask = get_valid(annot_pl, ignore_label)
        # with tf.variable_scope("PredIou"):
        #     (pred_miou, pred_conf_mat, pred_update), _ = get_miou(not_correct, pred_tensor, num_classes, ignore_label, do_ood, neg_validity_mask)
        with tf.variable_scope("DistIou"):
            (dist_miou, dist_conf_mat, dist_update), _ = get_miou(not_correct, dist_class, num_classes, ignore_label, do_ood, neg_validity_mask)

        weights = tf.to_float(neg_validity_mask)

        num_thresholds = 200

        with tf.variable_scope("Roc"):
            RocPoints, roc_update = tf.contrib.metrics.streaming_curve_points(not_correct,dist_pred,weights,num_thresholds,curve='ROC')
        with tf.variable_scope("Pr"):
            PrPoints, pr_update = tf.contrib.metrics.streaming_curve_points(not_correct,dist_pred,weights,num_thresholds,curve='PR')

        dbg = []#[not_correct, dist_pred, dist_class]

    stream_vars_valid = [v for v in tf.local_variables() if 'Roc/' in v.name]
    reset_op = tf.variables_initializer(stream_vars_valid)

    update_op = [dist_update]
    if not FLAGS.write_out:
        update_op += [pr_update, roc_update]
    update_op = tf.group(update_op)

    mean = np.reshape(mean, mean_p.get_shape().as_list())
    var_inv = np.reshape(var_inv, var_inv_p.get_shape().as_list())

    input_fetch = [input_name, input_tensor, annot_tensor]

    fetch = {"update": update_op}

    if FLAGS.train_kernel:
        fetch["predictions"] = pred_tensor
        fetch["min_dist_out"] = min_dist[0]

    if FLAGS.write_img:
        fetch["prediction_colour"] = pred_colour
        fetch["dist_out"] = tf.cast(dist_colour[0], tf.uint8)
        fetch["full_dist_out"] = full_dist[0]
        fetch["min_dist_out"] = min_dist[0]

    if FLAGS.write_out:
        fetch["img_dist_out"] = img_dist[0]
        fetch["unscaled_logits_out"] = unscaled_logits[0]

    grads = tf.gradients(min_dist, placeholder_tensor)
    epsilon_pl = tf.placeholder(tf.float32, (), "epsilon")
    if epsilon > 0.0:
        adv_img = placeholder_tensor - epsilon_pl*tf.sign(grads)
    else:
        adv_img = tf.expand_dims(placeholder_tensor, 0)

    num_step = num_images // batch
    print("running for", num_step, "steps")
    #os.makedirs(dump_dir, exist_ok=True)

    if FLAGS.write_out:
        write_queue = Queue(30)
        num_writers = 20
        writers = [ParallelWriter(write_queue) for i in range(num_writers)]

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()],
                    {mean_p: mean, var_inv_p: var_inv})
        tf.train.start_queue_runners(sess)
        vars_toload = [v for v in tf.global_variables() if v not in vars_noload]
        saver = tf.train.Saver(vars_toload)
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
            dump_filename = os.path.join(dump_dir, filename + ".npy")
            adv_img_out = sess.run(adv_img, feed_dict={placeholder_tensor: img_raw, annot_pl: annot_raw, epsilon_pl: epsilon})
            adv_img_out = adv_img_out[0]

            #import pdb; pdb.set_trace()
            #sess.run(reset_op)
            res, dbg_v = sess.run([fetch, dbg], feed_dict={placeholder_tensor: adv_img_out, annot_pl: annot_raw})

            roc = sess.run(RocPoints)
            auc = -np.trapz(roc[:,1], roc[:,0])
            # if auc <= 0.8480947:
            #     ###DBG
            #     def sigmoid(x):
            #         return 1 / (1 + np.exp(-x))
            #     min_og, not_correct_out = sess.run([min_dist, not_correct], {placeholder_tensor: img_raw, annot_pl: annot_raw})
            #     min_adv = sess.run(min_dist, {placeholder_tensor: adv_img_out, annot_pl: annot_raw})
            #     norm_min_og = (min_og - mean_value)/std_value
            #     norm_min_adv = (min_adv - mean_value)/std_value
            #     pred_og = sigmoid(norm_min_og * 0.13273349404335022 + 0.38076120615005493)
            #     pred_adv = sigmoid(norm_min_adv * 0.13273349404335022 + 0.38076120615005493)
            #     pred_avg = np.mean([pred_og, pred_adv],0)
            #     #######
            #     import pdb; pdb.set_trace()
            #     print(filename)

            dist_miou_v = sess.run([dist_miou])

            if FLAGS.train_kernel:
                predictions = res["predictions"]
                min_dist_out = res["min_dist_out"]
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

            if FLAGS.write_img:
                prediction_colour = res["prediction_colour"]
                dist_out = res["dist_out"]
                full_dist_out = res["full_dist_out"]
                predictions = res["predictions"]
                min_dist_out = res["min_dist_out"]

                # annot_out = res[8][0]
                # n_values = np.max(annot_out) + 1
                # one_hot_out = np.eye(n_values)[annot_out][...,0,:num_classes]

                min_dist_v = min_dist_out# np.expand_dims(np.nanmin(full_dist_out, -1), -1)
                min_dist_v[np.logical_not(np.isfinite(min_dist_v))] = np.nanmin(min_dist_out)
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
                write_queue.put((idx, save_location, prediction_colour))
                write_queue.put((idx, min_filename, min_dist_v))
                write_queue.put((idx, dist_filename, dist_out))

            if FLAGS.write_out:
                img_dist_out = res["img_dist_out"]
                unscaled_logits_out = res["unscaled_logits_out"]

                #if not os.path.exists(dump_filename):
                write_queue.put((idx, dump_filename, {"dist": img_dist_out, "unscaled_logits": unscaled_logits_out}))
                #else:
                #    print("skipping", filename, "                          ")

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

            elapsed = timeit.default_timer() - start_time
            end = "\r"
            if idx % 50 == 0:
                #every now and then do regular print
                end = "\n"
            if FLAGS.write_out:
                qsize = write_queue.qsize()
            else:
                qsize = 0

            print('{0:.4f} iter: {1}, iou: {2:.6f}, auc: {3:.6f}'.format(elapsed, idx+1, dist_miou_v[0], auc))

        if not FLAGS.write_out:
            roc = sess.run(RocPoints)
            pr = sess.run(PrPoints)

            make_plots(roc,pr,num_thresholds)

        if FLAGS.write_out:
            for w in writers:
                w.close()
        print('{0:.4f} iter: {1}, iou: {2:.6f}'.format(elapsed, idx+1, dist_miou_v[0]))




def main(_):
    #test_plots()
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
        if FLAGS.write_out or FLAGS.use_train:
            input_reader = pipeline_config.ood_train_input_reader
        else:
            input_reader = pipeline_config.ood_eval_input_reader
    else:
        if FLAGS.use_train:
            input_reader = pipeline_config.train_input_reader
        else:
            input_reader = pipeline_config.eval_input_reader

    input_reader.shuffle = True
    input_reader.num_epochs = 1
    input_reader.num_examples = min(1500, input_reader.num_examples)
    input_dict = dataset_builder.build(input_reader)

    ignore_label = pipeline_config.ood_config.ignore_label

    run_inference_graph(segmentation_model, FLAGS.trained_checkpoint,
                        input_dict, input_reader.num_examples, ignore_label, input_shape, pad_to_shape,
                        label_map, output_directory, num_classes, eval_dir, min_dir, dist_dir, hist_dir,
                        dump_dir)

if __name__ == '__main__':
    tf.app.run()
