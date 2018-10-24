from functools import partial
import tensorflow as tf
from protos import losses_pb2


def _softmax_classification_loss(predictions, labels, ignore_label):
    flattened_labels = tf.reshape(labels, shape=[-1])
    num_classes = predictions.get_shape().as_list()[-1]
    predictions = tf.reshape(predictions, [-1, num_classes])

    one_hot_target = tf.contrib.slim.one_hot_encoding(
                            tf.cast(flattened_labels, tf.int32),
                            num_classes, on_value=1.0, off_value=0.0)
    not_ignore_mask = tf.to_float(
                tf.not_equal(flattened_labels, ignore_label))

    return tf.losses.softmax_cross_entropy(
                    one_hot_target,
                    logits=tf.to_float(predictions),
                    weights=not_ignore_mask)

def _focal_loss(logits, labels, ignore_label, gamma=2.0, alpha=0.25):
    """
    focal loss for multi-classification
    FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
    Notice: logits is probability after softmax
    gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
    d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
    Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
    Focal Loss for Dense Object Detection, 130(4), 485–491.
    https://doi.org/10.1016/j.ajodo.2005.02.022
    :param labels: ground truth labels, shape of [batch_size]
    :param logits: model's output, shape of [batch_size, num_cls]
    :param gamma:
    :param alpha:
    :return: shape of [batch_size]
    """
    epsilon = 1.e-9
    # labels = tf.to_int64(labels)
    # labels = tf.convert_to_tensor(labels, tf.int64)
    # logits = tf.convert_to_tensor(logits, tf.float32)
    num_classes = logits.get_shape().as_list()[-1]
    one_hot_target = tf.one_hot(tf.cast(labels, tf.int32),
                            num_classes, on_value=1.0, off_value=0.0)
    one_hot_target = tf.squeeze(one_hot_target, axis=[3])
    model_out = logits #tf.add(logits, epsilon)
    ce = tf.multiply(one_hot_target, -tf.log(tf.clip_by_value(model_out,epsilon,1.0)))
    weight = tf.multiply(one_hot_target, tf.pow(tf.subtract(1., model_out), gamma))
    fl = tf.multiply(alpha, tf.multiply(weight, ce))
    reduced_fl = tf.reduce_max(fl, axis=-1, keepdims=True)
    not_ignore_mask = tf.to_float(
                tf.not_equal(labels, ignore_label))
    # reduced_fl = tf.reduce_sum(fl, axis=1)  # same as reduce_max
    return tf.reduce_mean(reduced_fl * not_ignore_mask)


def _l2norm(predictions, labels, ignore_label):
    return 0.5 * tf.reduce_mean(tf.squared_difference(labels, predictions))

def _l2norm_mag(predictions, labels, ignore_label):
    sq_dif = tf.squared_difference(labels, predictions)
    
    eps = 0.1
    mag = tf.sqrt(tf.squared_difference(labels[:,:,:,0], labels[:,:,:,1]))
    mag = tf.expand_dims(mag, -1)
    weight = tf.cast(1/(mag + eps), dtype=tf.float32) #10 -> 0

    loss = 0.5 * tf.reduce_mean(sq_dif * weight)
    
    return loss

def _l2norm_smooth(predictions, labels, ignore_label):
    diff = 0.5 * tf.reduce_mean(tf.squared_difference(labels, predictions))

    
    hor = tf.squared_difference(predictions[:, 1:, :, :], predictions[:, :-1, :, :])
    vert = tf.squared_difference(predictions[:, :, 1:, :], predictions[:, :, :-1, :])
    diag = tf.squared_difference(predictions[:, 1:, 1:, :], predictions[:, :-1, :-1, :])
    anti_diag = tf.squared_difference(predictions[:, 1:, -1:, :], predictions[:, :-1, 1:, :])

    smooth = 0.5 * tf.add_n([tf.reduce_mean(v) for v in [hor, vert, diag, anti_diag]])

    return diff + (0.01 * smooth)

def build(loss_config):
    if not isinstance(loss_config, losses_pb2.Loss):
        raise ValueError('loss_config not of type '
                         'losses_pb2.ClassificationLoss.')

    class_loss = None

    class_loss_type = loss_config.classification_loss.WhichOneof('loss_type')
    if class_loss_type == 'softmax':
        class_loss = partial(_softmax_classification_loss,
            ignore_label=loss_config.ignore_label)
    elif class_loss_type == 'focal':
        class_loss = partial(_focal_loss,
            ignore_label=loss_config.ignore_label)
    else:
        raise ValueError('Empty class loss config.')
    
    return class_loss