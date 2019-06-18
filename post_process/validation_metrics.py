import tensorflow as tf
import numpy as np
import sklearn as sk
import cv2

num_thresholds = 400
eps = 1e-7

apply_filter = False

def filter_ood(imgs, thresh=180, dilate=5, erode=5):
    if dilate % 2 == 0:
        dilate -= 1
    if erode % 2 == 0:
        erode -= 1
    if dilate < 1:
        dilate = 1
    if erode < 1:
        erode = 1
    # import pdb; pdb.set_trace()
    all_probs = []
    for img in imgs:
        img -= img.min()
        img /= img.max()
        edges = cv2.Canny((img*255).astype(np.uint8), thresh, thresh)
        di = cv2.dilate(edges, np.ones((dilate, dilate)))
        er = cv2.erode(di, np.ones((erode, erode)))

        dtform = cv2.distanceTransform(255 - er,
                        distanceType=cv2.DIST_L2,
                        maskSize=cv2.DIST_MASK_PRECISE)

        dtform[dtform > 10] = 10
        dtform /= 10

        border_probs = dtform
        probs = cv2.dilate(border_probs*img, np.ones((3,3)))
        # import matplotlib.pyplot as plt
        # plt.subplot(2,1,1)
        # plt.imshow(img)
        # plt.subplot(2,1,2)
        # plt.imshow(probs)
        # plt.show()
        all_probs.append(probs)
    return np.array(all_probs)

def get_metric_ops(annot, prediction, weights):

    new_pred = prediction
    if apply_filter:
        new_pred = tf.py_func(filter_ood, [prediction], tf.float32, stateful=False)
        new_pred.set_shape(prediction.shape)

    res, update = tf.contrib.metrics.precision_recall_at_equal_thresholds(tf.cast(annot, tf.bool),new_pred,weights,num_thresholds, name="ConfMat")

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