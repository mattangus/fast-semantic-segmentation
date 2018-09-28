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

def _focal_loss(labels, logits, ignore_label, gamma=2.0, alpha=0.25):
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
    one_hot_target = tf.contrib.slim.one_hot_encoding(
                            tf.cast(labels, tf.int32),
                            num_classes, on_value=1.0, off_value=0.0)
    model_out = tf.add(logits, epsilon)
    ce = tf.multiply(one_hot_target, -tf.log(model_out))
    weight = tf.multiply(one_hot_target, tf.pow(tf.subtract(1., model_out), gamma))
    fl = tf.multiply(alpha, tf.multiply(weight, ce))
    reduced_fl = tf.reduce_max(fl, axis=-1)
    not_ignore_mask = tf.to_float(
                tf.not_equal(labels, ignore_label))
    # reduced_fl = tf.reduce_sum(fl, axis=1)  # same as reduce_max
    return reduced_fl * not_ignore_mask

def _l2norm(predictions, labels, ignore_label):
    return 0.5 * tf.reduce_mean(tf.squared_difference(labels, predictions))

def _l2norm_mag(predictions, labels, ignore_label):
    sq_dif = tf.squared_difference(labels, predictions)



def _l2norm_smooth(predictions, labels, ignore_label):
    l2norm_term = _l2norm(predictions, labels, ignore_label)
    raise NotImplementedError("l2 norm smooth is not implemented")

def build(loss_config):
    if not isinstance(loss_config, losses_pb2.Loss):
        raise ValueError('loss_config not of type '
                         'losses_pb2.ClassificationLoss.')

    class_loss = None
    vec_loss = None

    class_loss_type = loss_config.classification_loss.WhichOneof('loss_type')
    if class_loss_type == 'softmax':
        class_loss = partial(_softmax_classification_loss,
            ignore_label=loss_config.ignore_label)
    elif class_loss_type == 'focal':
        class_loss = partial(_focal_loss,
            ignore_label=loss_config.ignore_label)
    else:
        raise ValueError('Empty class loss config.')

    if loss_config.use_vec_loss:
        vec_loss_type = loss_config.vector_field_loss.WhichOneof('loss_type')
        if vec_loss_type == 'l2norm':
            vec_loss = partial(_l2norm,
                ignore_label=loss_config.ignore_label)
        elif vec_loss_type == 'l2normmag':
            vec_loss = partial(_l2norm_mag,
                ignore_label=loss_config.ignore_label)
        elif vec_loss_type == 'l2normsmooth':
            vec_loss = partial(_l2norm_smooth,
                ignore_label=loss_config.ignore_label)
        else:
            raise ValueError('Empty vec loss config.')
    
    return class_loss, vec_loss