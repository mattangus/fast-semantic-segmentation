import tensorflow as tf
import numpy as np


num_thresholds = 200

def get_metric_ops(annot, prediction, weights):
    with tf.variable_scope("Roc"):
        RocPoints, roc_update = tf.contrib.metrics.streaming_curve_points(annot,prediction,weights,num_thresholds,curve='ROC')
    with tf.variable_scope("Pr"):
        PrPoints, pr_update = tf.contrib.metrics.streaming_curve_points(annot,prediction,weights,num_thresholds,curve='PR')

    update = tf.group([roc_update, pr_update])
    metrics = {
        "roc": RocPoints,
        "pr": PrPoints,
    }

    return metrics, update

def _auc(points):
    return -np.trapz(points[:,1], points[:,0])

def get_metric_values(metrics):
    roc = metrics["roc"]
    pr = metrics["pr"]

    auroc = _auc(roc)
    aupr = _auc(pr)

    fpr_tpr = sorted(roc, key=lambda x: np.abs(x[1] - 0.95))
    fpr_at_tpr = fpr_tpr[0][0]

    detection_error = 0.5*(1 - fpr_tpr[0][1]) + 0.5*fpr_at_tpr

    ret = {
        "auroc": auroc,
        "aupr": aupr,
        "fpr_at_tpr": fpr_at_tpr,
        "detection_error": detection_error,
    }

    return ret