from functools import partial
import tensorflow as tf
import numpy as np
from protos import losses_pb2

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

def _max_dist_batch(logits, labels, ignore_label):
    num_classes = logits.get_shape().as_list()[-1]
    one_hot = tf.squeeze(tf.one_hot(tf.cast(labels, tf.int32), num_classes), 3)
    resized = tf.image.resize_nearest_neighbor(one_hot, logits.get_shape().as_list()[1:-1])
    sorted_feats = logits*resized
    means = tf.reduce_mean(sorted_feats, axis=0, keepdims=True) #covs = _moments(sorted_feats)
    loss = -tf.reduce_mean(tf.square(tf.expand_dims(means, -1) - tf.expand_dims(means, -2)))
    return loss

def build(loss_config):
    if not isinstance(loss_config, losses_pb2.Loss):
        raise ValueError('loss_config not of type '
                         'losses_pb2.ClassificationLoss.')
    
    dist_loss = None

    dist_loss_type = loss_config.dist_loss.WhichOneof('loss_type')
    if dist_loss_type == 'batch':
        dist_loss = partial(_max_dist_batch,
            ignore_label=loss_config.ignore_label)
    elif dist_loss_type == 'none':
        pass # no distance loss
    else:
        raise ValueError('Empty dist loss config.')
    
    return dist_loss