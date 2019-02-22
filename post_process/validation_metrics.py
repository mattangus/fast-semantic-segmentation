import tensorflow as tf
import numpy as np
import sklearn as sk

num_thresholds = 400
eps = 1e-7

def get_metric_ops(annot, prediction, weights):

    res, update = tf.contrib.metrics.precision_recall_at_equal_thresholds(tf.cast(annot, tf.bool),prediction,weights,num_thresholds, name="ConfMat")

    tp = res.tp
    fp = res.fp
    tn = res.tn
    fn = res.fn

    with tf.variable_scope("Roc"):
        tpr = tp / tf.maximum(eps, tp + fn)
        fpr = fp / tf.maximum(eps, fp + tn)
        RocPoints = tf.stack([fpr, tpr], -1)
    
    with tf.variable_scope("Pr"):
        PrPoints = tf.stack([res.recall, res.precision], -1)

    with tf.variable_scope("iou"):
        denom = tf.maximum(eps, tp + fn + fp)
        IouPoints = tp / denom

    metrics = {
        "roc": RocPoints,
        "pr": PrPoints,
        "iou": IouPoints,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }

    return metrics, update

def _auc(points):
    return -np.trapz(points[:,1], points[:,0])

def get_metric_values(metrics):
    roc = metrics["roc"]
    pr = metrics["pr"]
    iou = metrics["iou"]

    # import matplotlib.pyplot as plt
    # import pdb; pdb.set_trace()

    auroc = _auc(roc)
    aupr = _auc(pr)
    maxiou = np.max(iou)

    prev = np.array([0.,0.])
    fpr_at_tpr_p = None
    for p in reversed(roc):
        if p[1] >= 0.95:
            pct = (0.95 - prev[1]) / (p[1] - prev[1])
            fpr_at_tpr_p = pct * p + (1 - pct) * prev
            break
        prev = p
    
    if fpr_at_tpr_p is None:
        fpr_at_tpr = 1.0
        detection_error = 1.0
    else:
        fpr_at_tpr = fpr_at_tpr_p[0]
        detection_error = 0.5*(1 - fpr_at_tpr_p[1]) + 0.5*fpr_at_tpr_p[0]

    tp = metrics["tp"]
    fp = metrics["fp"]
    tn = metrics["tn"]
    fn = metrics["fn"]

    ret = {
        "auroc": auroc,
        "aupr": aupr,
        "max_iou": maxiou,
        "fpr_at_tpr": fpr_at_tpr,
        "detection_error": detection_error,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }

    return ret