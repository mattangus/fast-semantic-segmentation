from functools import partial
import tensorflow as tf
import numpy as np
from protos import losses_pb2

DEBUG = []

def _moments(x):
    static_shape = x.get_shape().as_list()
    dyn_shape = tf.shape(x)
    if None in static_shape:
        target_shape = [dyn_shape[0], tf.reduce_prod(dyn_shape[1:-1]), -1]
        final_shape = [1, dyn_shape[1:]]
        n_samp = dyn_shape[0]
    else:
        target_shape = [static_shape[0], np.prod(static_shape[1:-1]), -1]
        final_shape = [1, static_shape[1:]]
        n_samp = static_shape[0]
    
    x = tf.reshape(x, target_shape)
    mean_x = tf.reduce_mean(x, axis=0, keepdims=True)
    mx = tf.matmul(tf.expand_dims(mean_x,-1), tf.expand_dims(mean_x,-2))
    vx = tf.reduce_mean(tf.matmul(tf.expand_dims(x,-1), tf.expand_dims(x,-2)), axis=0, keepdims=True)
    cov_xx = vx - mx
    return tf.reshape(mean_x, final_shape), tf.reshape(cov_xx, final_shape)

def safe_f(x, f, safe_f=tf.zeros_like, safe_x=tf.ones_like, cmp=lambda x: tf.not_equal(x, 0.)):
    x_ok = cmp(x)
    safe_x = tf.where(x_ok, x, safe_x(x))
    return tf.where(x_ok, f(safe_x), safe_f(x))

def _l2_dist_batch(logits, labels, ignore_label, num_classes):
    one_hot = tf.squeeze(tf.one_hot(tf.cast(labels, tf.int32), num_classes), 3)
    #TODO: try bilinear resize!
    resized = tf.expand_dims(tf.image.resize_nearest_neighbor(one_hot, logits.get_shape().as_list()[1:-1]),-2)
    sorted_feats = tf.expand_dims(logits, -1)*resized
    means = tf.reduce_sum(sorted_feats, axis=0) * safe_f(tf.reduce_sum(resized,0), tf.reciprocal) #covs = _moments(sorted_feats)
    n = 1.0
    #eye = tf.eye(num_classes)
    diffs = tf.expand_dims(means, -1) - tf.expand_dims(means, -2)

    sq_dist = tf.square(diffs)
    dot = tf.reduce_sum(sq_dist, -3)
    dist = safe_f(dot, tf.sqrt, safe_x=tf.zeros_like, cmp=lambda x: tf.greater(x,0.))
    #lin_inv = tf.maximum(0., 100. - dist)
    dist_inv = safe_f(dist, tf.reciprocal)
    dist_inv = tf.maximum(dist_inv, tf.ones_like(dist_inv) * (1/1000))
    # eps = tf.where(tf.equal(dist, 0), tf.ones_like(dist), tf.zeros_like(dist))
    # safe_dist = tf.where(gt, dist, tf.ones_like(dist))
    # inv_dist = tf.reciprocal(safe_dist)
    # inv_dist = tf.where(gt, inv_dist, tf.zeros_like(dist))
    # safe_dist = tf.where(gt, dist, tf.ones_like(dist))
    # tf.where(gt, tf.reciprocal(safe_dist), safe_f(x))
    # inv_dist = safe_reciprocal(dist)
    #inv_dist = tf.where(tf.is_finite(inv_dist), inv_dist, tf.zeros_like(inv_dist))
    loss = n * tf.reduce_sum(dist_inv, -1)
    # global DEBUG
    # DEBUG = [diffs, sq_dist, dist, loss]
    return tf.reduce_mean(loss)

def _mahal_dist_batch(logits, labels, ignore_label, num_classes):
    one_hot = tf.squeeze(tf.one_hot(tf.cast(labels, tf.int32), num_classes), 3)
    #TODO: try bilinear resize!
    resized = tf.expand_dims(tf.image.resize_nearest_neighbor(one_hot, logits.get_shape().as_list()[1:-1]),-2)
    sorted_feats = tf.expand_dims(logits, -1)*resized
    means = tf.reduce_sum(sorted_feats, axis=[0,1,2]) * safe_f(tf.reduce_sum(resized,[0,1,2]), tf.reciprocal) #covs = _moments(sorted_feats)
    mean_sub = tf.reduce_mean(sorted_feats - tf.expand_dims(means,0),[0,1,2])
    cov = tf.reduce_mean(tf.matmul(tf.expand_dims(mean_sub,-1), tf.expand_dims(mean_sub,-2)))
    n = 1.0
    #eye = tf.eye(num_classes)
    diffs = tf.expand_dims(means, -1) - tf.expand_dims(means, -2)
    # import pdb; pdb.set_trace()
    # tpose = tf.transpose(diffs, [1,2,0])
    # covs = tf.matmul(tf.expand_dims(tpose,-1), tf.expand_dims(tpose,-2))
    # mags = tf.mean()

    sq_dist = tf.square(diffs)
    dot = tf.reduce_sum(sq_dist, -3)
    dist = safe_f(dot, tf.sqrt, safe_x=tf.zeros_like, cmp=lambda x: tf.greater(x,0.))
    #lin_inv = tf.maximum(0., 100. - dist)
    dist_inv = safe_f(dist, tf.reciprocal)
    dist_inv = tf.maximum(dist_inv, tf.ones_like(dist_inv) * (1/1000))
    # eps = tf.where(tf.equal(dist, 0), tf.ones_like(dist), tf.zeros_like(dist))
    # safe_dist = tf.where(gt, dist, tf.ones_like(dist))
    # inv_dist = tf.reciprocal(safe_dist)
    # inv_dist = tf.where(gt, inv_dist, tf.zeros_like(dist))
    # safe_dist = tf.where(gt, dist, tf.ones_like(dist))
    # tf.where(gt, tf.reciprocal(safe_dist), safe_f(x))
    # inv_dist = safe_reciprocal(dist)
    #inv_dist = tf.where(tf.is_finite(inv_dist), inv_dist, tf.zeros_like(inv_dist))
    loss = n * tf.reduce_sum(dist_inv, -1)
    # global DEBUG
    # DEBUG = [diffs, sq_dist, dist, loss]
    # import pdb; pdb.set_trace()
    return tf.reduce_mean(loss) + tf.reduce_mean(cov)

def build(loss_config, num_classes):
    if not isinstance(loss_config, losses_pb2.Loss):
        raise ValueError('loss_config not of type '
                         'losses_pb2.ClassificationLoss.')
    
    dist_loss = None

    dist_loss_type = loss_config.dist_loss.WhichOneof('loss_type')
    if dist_loss_type == 'l2':
        dist_loss = partial(_l2_dist_batch,
            ignore_label=loss_config.ignore_label, num_classes=num_classes)
    if dist_loss_type == 'mahal':
        dist_loss = partial(_mahal_dist_batch,
            ignore_label=loss_config.ignore_label, num_classes=num_classes)
    elif dist_loss_type == 'none':
        pass # no distance loss
    else:
        raise ValueError('Empty dist loss config.')
    
    return dist_loss