from functools import partial
import tensorflow as tf
from protos import losses_pb2

import helpers

epsilon = 1.e-5

def _softmax_classification_loss(predictions, labels, ignore_label):
    #flattened_labels = tf.reshape(labels, shape=[-1])
    num_classes = predictions.get_shape().as_list()[-1]
    #predictions = tf.reshape(predictions, [-1, num_classes])
    #predictions = tf.to_float(predictions)

    one_hot_target = tf.contrib.slim.one_hot_encoding(
                            tf.cast(labels, tf.int32),
                            num_classes, on_value=1.0, off_value=0.0)
    one_hot_target = tf.squeeze(one_hot_target, -2)
    # not_ignore_mask = tf.reduce_sum(one_hot_target, -1)
    ne = [tf.not_equal(labels, il) for il in ignore_label]
    ne = [tf.to_float(v) for v in ne]
    # ne_sums = [tf.reduce_sum(v) for v in ne]
    not_ignore_mask = ne.pop(0)
    for v in ne:
        not_ignore_mask = tf.multiply(not_ignore_mask, v)
    not_ignore_mask = tf.squeeze(not_ignore_mask, -1)
    #tf.summary.image("not_ignore_mask", tf.reshape(not_ignore_mask, tf.shape(labels)))

    #uncomment the follwoing to debug if one_hot contains invalid values
    #(i.e. a vec like [0,0,0,...]) the value of num_act should be 0

    # ignore_mask = 1 - not_ignore_mask #tf.to_float(tf.logical_not(ne))
    # eq_count = tf.reduce_sum(ignore_mask)
    # ne_count = tf.reduce_sum(not_ignore_mask)
    # max_val = tf.reduce_max(flattened_labels*not_ignore_mask)
    # not_max_val = tf.reduce_max(flattened_labels*ignore_mask)
    # z_max_val = tf.reduce_max(flattened_labels*ignore_mask*not_ignore_mask)
    # num_act = tf.reduce_sum(tf.to_float(tf.equal(tf.reduce_sum(one_hot_target, -1), 0)) * not_ignore_mask)
    # one_hot_target = tf.Print(one_hot_target,
    #         ["num_act:", num_act,
    #          "eq_count:", eq_count,
    #          "ne_count:", ne_count,
    #          "max:", max_val,
    #          "zero:", z_max_val,
    #          "not_max", not_max_val,
    #          "ig_max", tf.reduce_max(ignore_mask),
    #          "nig_max", tf.reduce_max(not_ignore_mask)])

    # c_ent = tf.nn.softmax_cross_entropy_with_logits_v2(
    #     labels=one_hot_target,
    #     logits=predictions)*not_ignore_mask
    c_ent = tf.losses.softmax_cross_entropy(
                    one_hot_target,
                    logits=predictions,
                    weights=not_ignore_mask,
                    loss_collection=None,
                    reduction="none")

    # p = tf.print([tf.reduce_max(predictions), tf.reduce_min(predictions)])
    # with tf.control_dependencies([p]):
    predictions = tf.debugging.assert_all_finite(predictions,"predictions nan")
    smax = tf.nn.softmax(predictions)
    smax = tf.debugging.assert_all_finite(smax,"smax nan")
    max_pred = tf.reduce_max(predictions, -1, keepdims=True)
    max_sub = predictions - max_pred
    log_pred = (max_sub - tf.log(tf.reduce_sum(tf.exp(max_sub), -1, keepdims=True)+epsilon))
    log_pred = tf.debugging.assert_all_finite(log_pred,"log_pred nan")


    lam =  0.0001
    ent_loss = -tf.reduce_mean(smax*log_pred, -1)*lam
    #ent_loss = -tf.reduce_mean(tf.log(smax + epsilon)*smax, -1)*lam

    ent_loss = tf.debugging.assert_all_finite(ent_loss,"ent_loss nan")
    
    #if any equal 254 the whole image uses ent_loss
    weights = tf.to_float(tf.reduce_sum(tf.to_float(tf.squeeze(tf.equal(labels, 254),-1)),[1,2],keepdims=True) > 0)

    # p = tf.print([tf.squeeze(weights)])
    # with tf.control_dependencies([p]):
    # import pdb; pdb.set_trace()
    return tf.reduce_mean(tf.div_no_nan(tf.reduce_sum(ent_loss*weights + (1 - weights)*c_ent,[1,2]),tf.reduce_sum(not_ignore_mask,[1,2])))
    #return tf.reduce_mean(tf.div_no_nan(tf.reduce_sum(c_ent,[1,2]),tf.reduce_sum(not_ignore_mask,[1,2])))

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

def _confidence_loss(predictions, labels, ignore_label):
    #take the last channel of preds for pixel confidence
    confidence = tf.nn.sigmoid(predictions[...,-1:])

    #take the rest for predictions
    predictions = tf.nn.softmax(predictions[...,:-1])
    num_classes = predictions.shape.as_list()[-1]

    labels = tf.cast(labels, tf.int32)

    eps = 1e-12
    clamp_pred = tf.clip_by_value(predictions, 0. + eps, 1. - eps,)
    clamp_conf = tf.clip_by_value(confidence, 0. + eps, 1. - eps,)

    # Randomly set half of the confidences to 1 (i.e. no hints)
    b = tf.math.round(tf.random.uniform(clamp_conf.shape))
    conf_rand = clamp_conf * b + (1 - b)
    one_hot = tf.one_hot(tf.squeeze(labels,-1), num_classes)
    pred_new = clamp_pred * conf_rand + one_hot * (1 - conf_rand)
    pred_new = tf.log(pred_new)

    weights = tf.to_float(helpers.get_valid(labels, ignore_label))

    xentropy_loss = -tf.reduce_sum(pred_new * one_hot, -1, keepdims=True) * weights
    confidence_loss = -tf.log(confidence) * weights

    total_loss = tf.reduce_mean(xentropy_loss) + (0.5 * tf.reduce_mean(confidence_loss))

    return total_loss

def build(loss_config, ignore_label):
    if not isinstance(loss_config, losses_pb2.Loss):
        raise ValueError('loss_config not of type '
                         'losses_pb2.ClassificationLoss.')

    class_loss = None

    class_loss_type = loss_config.classification_loss.WhichOneof('loss_type')
    if class_loss_type == 'softmax':
        class_loss = partial(_softmax_classification_loss,
            ignore_label=ignore_label)
    elif class_loss_type == 'focal':
        class_loss = partial(_focal_loss,
            ignore_label=ignore_label)
    elif class_loss_type == "confidence":
        class_loss = partial(_confidence_loss,
            ignore_label=ignore_label)
    else:
        raise ValueError('Empty class loss config.')
    
    return class_loss