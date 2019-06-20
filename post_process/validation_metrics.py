import tensorflow as tf
import numpy as np
import sklearn as sk
import cv2

num_thresholds = 400
eps = 1e-7

thresh_list = [110, 120, 130, 140, 150, 160, 170]

def filter_ood(imgs, thresh=110, dilate=5, erode=5):
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
        probs = cv2.dilate(border_probs*np.squeeze(img), np.ones((3,3)))
        # import matplotlib.pyplot as plt
        # plt.subplot(2,1,1)
        # plt.imshow(img)
        # plt.subplot(2,1,2)
        # plt.imshow(probs)
        # plt.show()
        # import pdb; pdb.set_trace()
        all_probs.append(probs)
    return np.array(all_probs)

def get_metric_ops(annot, prediction, weights):
    metrics, update = get_metric_ops_thresh(annot, prediction, weights)
    update = [update]

    all_metrics = {}
    for thresh in thresh_list:
        with tf.variable_scope("thresh_" + str(thresh)):
            cur_metric, cur_update = get_metric_ops_thresh(annot, prediction, weights, thresh)
            all_metrics[thresh] = cur_metric
            update.append(cur_update)

    return metrics, tf.group(update), all_metrics


def get_metric_ops_thresh(annot, prediction, weights, filter_value=None):

    new_pred = prediction
    if filter_value is not None:
        new_pred = tf.py_func(filter_ood, [prediction, filter_value], tf.float32, stateful=False)
        new_pred.set_shape(prediction.shape)
        new_pred = tf.clip_by_value(new_pred,0,1)

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
        "pred": prediction,
        "new_pred": new_pred
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

counter = 0

def get_best_metric_values(all_metrics, original_results):
    # global counter
    all_results = {}

    best_t = None
    best_num_better = 0

    lt = lambda x, y: x < y
    gt = lambda x, y: x > y

    for t in all_metrics:
        cur_metric = all_metrics[t]
        cur_results = get_metric_values(cur_metric)
        all_results[t] = cur_results

        num_better = 0
        for name, comp in [("auroc", gt), ("aupr", gt), ("max_iou", gt), ("fpr_at_tpr", lt), ("detection_error", lt)]:
            if comp(cur_results[name], original_results[name]):
                num_better += 1
        if best_t is None:
            best_t = t
            best_num_better = num_better
        elif best_num_better < num_better:
            best_t = t
            best_num_better = num_better

    for name, comp in [("auroc", gt), ("aupr", gt), ("max_iou", gt), ("fpr_at_tpr", lt), ("detection_error", lt)]:
        print(name, all_results[best_t][name] - original_results[name])

    # if best_num_better > 0:
    #     import pdb; pdb.set_trace()

    print("num better", best_num_better, "best_t", best_t)

    return all_results[best_t]