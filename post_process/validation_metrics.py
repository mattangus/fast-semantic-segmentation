import tensorflow as tf
import numpy as np
import sklearn as sk

num_thresholds = 500
eps = 1e-7

def _disc(n):
    if n % 2 == 0:
        n -= 1
    n = int((n - 1)/2)
    y,x = np.ogrid[-n: n+1, -n: n+1]
    mask = x**2+y**2 <= n**2
    return mask.astype(np.float32)

def _get_metric_from_res(res):
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
    return metrics

pr_values = tf.contrib.metrics.precision_recall_at_equal_thresholds

def get_metric_ops(annot, prediction, weights):

    with tf.variable_scope("test_metric"):
        kern = np.expand_dims(_disc(11), -1)
        if prediction._rank() != 4:
            prediction = tf.expand_dims(prediction,-1)
        erode = tf.nn.erosion2d(prediction, kern, [1,1,1,1], [1,1,1,1], "SAME")
        new_pred = tf.sqrt(tf.nn.dilation2d(erode, kern, [1,1,1,1], [1,1,1,1], "SAME")*prediction)
        test_res, test_update = pr_values(tf.cast(annot, tf.bool),new_pred,weights,num_thresholds, name="TestConfMat")
        test_metrics = _get_metric_from_res(test_res)

    res, update = pr_values(tf.cast(annot, tf.bool),prediction,weights,num_thresholds, name="ConfMat")

    metrics = _get_metric_from_res(res)

    return {"og": metrics, "test": test_metrics}, tf.group([update, test_update])

def _auc(points):
    return -np.trapz(points[:,1], points[:,0])

def _process_metrics(metrics):
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

def get_metric_values(metrics):

    test_metrics = metrics["test"]
    metrics = metrics["og"]

    ret = _process_metrics(metrics)

    test_ret = _process_metrics(test_metrics)

    ret["test_auroc"] = test_ret["auroc"]
    ret["test_aupr"] = test_ret["aupr"]
    ret["test_max_iou"] = test_ret["max_iou"]
    ret["test_fpr"] = test_ret["fpr_at_tpr"]
    ret["test_de"] = test_ret["detection_error"]

    return ret