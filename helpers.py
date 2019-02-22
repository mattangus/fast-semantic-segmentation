import numpy as np
import tensorflow as tf

def get_valid(labels, ignore_label):
    ne = [tf.not_equal(labels, il) for il in ignore_label]
    neg_validity_mask = ne.pop(0)
    for v in ne:
        neg_validity_mask = tf.logical_and(neg_validity_mask, v)
    
    return neg_validity_mask

def get_threshs(num_thresholds):
    eps = 1e-7
    #from https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/metrics/python/ops/metric_ops.py
    threshs = [(i + 1) * 1.0 / (num_thresholds - 1) for i in range(num_thresholds - 2)]
    threshs = [0.0 - eps] + threshs + [1.0 + eps]
    return threshs

def get_optimal_thresh(roc, threshs):
    #from http://www.medicalbiostatistics.com/roccurve.pdf page 6
    optimal = np.argmin(np.sqrt(np.square(1-roc[:,1]) + np.square(roc[:,0])))
    optimal_point = roc[optimal]
    optimal_thresh = threshs[optimal]
    return optimal_thresh, optimal_point