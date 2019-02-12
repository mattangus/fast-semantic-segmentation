import tensorflow as tf
import os
import numpy as np
from tensorflow.python.ops import metrics_impl

from . import post_processor as pp
from third_party.doc_inherit import doc_inherit
from helpers import get_valid
from . import validation_metrics as metrics
from libs.stat_computer import MeanComputer

def _safe_div(a,b):
    b = tf.ones_like(a)*b #broadcast
    return tf.where(tf.less(tf.abs(b), 1e-7), a, a/b)

class DropoutProcessor(pp.PostProcessor):
    
    def __init__(self, model, outputs_dict, num_classes,
                    annot, image, ignore_label, process_annot,
                    num_gpus, batch_size,
                    #class specific
                    num_runs):
        super().__init__("Dropout", model, outputs_dict, num_gpus)
        self.num_classes = num_classes
        self.annot = annot
        self.image = image
        self._num_runs = num_runs
        self.ignore_label = ignore_label
        self._process_annot = process_annot
        self._batch_size = batch_size

        self.pre_process_gpu = "gpu:0"
        if self.num_gpus > 1:
            self.pre_process_gpu = "gpu:1"

    def _dot_fn(self, x):
        return tf.squeeze(tf.matmul(tf.expand_dims(x,-2), tf.expand_dims(x,-1)),-1)

    @doc_inherit
    def post_process_ops(self):
        main_pred = self.outputs_dict[self.model.main_class_predictions_key]
        unscaled_logits = self.outputs_dict[self.model.unscaled_logits_key]
        pred_shape = main_pred.shape.as_list()

        #set shape based on runs and batch
        unscaled_logits.set_shape([self._num_runs * self._batch_size] + unscaled_logits.shape.as_list()[1:])
        self.annot.set_shape([self._batch_size] + self.annot.shape.as_list()[1:])

        #make predition
        logits = tf.image.resize_bilinear(unscaled_logits, pred_shape[1:3])
        pred = tf.nn.softmax(logits)
        stacked_pred = tf.stack(tf.split(pred, self._num_runs))

        self.mean_logits, self.all_variance = tf.nn.moments(stacked_pred, [0])
        self.prediction = tf.cast(tf.expand_dims(tf.argmax(self.mean_logits, -1),-1), tf.int32)
        self.interp_variance = tf.reduce_mean(self.all_variance, -1, keepdims=True)

        #get weights and new annot from predictions
        weights = tf.to_float(get_valid(self.annot, self.ignore_label))
        self.annot, self.num_classes = self._process_annot(self.annot, self.prediction, self.num_classes)

        #Popoviciu's inequality: var[X] <= (max - min)^2/4
        #https://stats.stackexchange.com/questions/45588/variance-of-a-bounded-random-variable
        max_var = 1./4.
        scale = 10
        max_var /= scale
        self.norm_variance = tf.nn.sigmoid((self.interp_variance - max_var) / max_var)

        with tf.device(self.pre_process_gpu):
            self.metrics, self.update = metrics.get_metric_ops(self.annot, self.norm_variance, weights)

    @doc_inherit
    def get_init_feed(self):
        return {}
        
    @doc_inherit
    def get_preprocessed(self):
        return tf.concat([self.image] * self._num_runs, axis=0)

    @doc_inherit
    def get_vars_noload(self):
        return []

    @doc_inherit
    def get_fetch_dict(self):
        fetch = []
        # for i in range(self._num_runs):
        #     # fetch.append({i: self.update_op})
        fetch.append({"res": self.norm_variance,})
        fetch.append({"update": self.update})
        fetch.append({"metrics": self.metrics})
        # fetch.append({"reset": self.reset_op})
        return fetch
    
    @doc_inherit
    def get_feed_dict(self):
        return {} #self.feed
    
    @doc_inherit
    def post_process(self, numpy_dict):
        # import matplotlib.pyplot as plt
        # plt.imshow(numpy_dict["res"][0,...,0])
        # plt.show()
        # import pdb; pdb.set_trace()
        results = metrics.get_metric_values(numpy_dict["metrics"])

        return results