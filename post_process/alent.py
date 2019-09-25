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

class AlEntProcessor(pp.PostProcessor):
    
    def __init__(self, model, outputs_dict, num_classes,
                    annot, image, path, ignore_label, process_annot,
                    num_gpus, batch_size,
                    #class specific
                    num_runs):
        super().__init__("AlEnt", model, outputs_dict, num_gpus)
        self.num_classes = num_classes
        self.annot = annot
        self.image = image
        self.path = path
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
        self.stacked_logits = tf.stack(tf.split(logits, self._num_runs))

        self.mean_logits, _ = tf.nn.moments(self.stacked_logits, [0])
        self.prediction = tf.cast(tf.expand_dims(tf.argmax(self.mean_logits, -1),-1), tf.int32)

        #get weights and new annot from predictions
        self.weights = tf.to_float(get_valid(self.annot, self.ignore_label))
        self.annot, self.num_classes = self._process_annot(self.annot, self.prediction, self.num_classes)

        smax = tf.nn.softmax(self.stacked_logits)
        max_pred = tf.reduce_max(self.stacked_logits, -1, keepdims=True)
        max_sub = self.stacked_logits - max_pred
        log_pred = (max_sub - tf.log(tf.reduce_sum(tf.exp(max_sub), -1, keepdims=True)+1.e-5))
        self.prediction = -tf.reduce_mean(smax*log_pred, [0,-1])

        #Popoviciu's inequality: var[X] <= (max - min)^2/4
        #https://stats.stackexchange.com/questions/45588/variance-of-a-bounded-random-variable
        max_var = 1./4.
        scale = 100
        max_var /= scale
        #self.norm_variance = tf.nn.sigmoid((self.interp_variance - max_var) / max_var)
        self.norm_variance = (self.prediction) / max_var

        with tf.device(self.pre_process_gpu):
            self.metrics, self.update, self.all_metrics = metrics.get_metric_ops(self.annot, self.prediction, self.weights)

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
        #fetch.append({"res": self.norm_variance,})
        fetch.append({"update": self.update})
        fetch.append({"metrics": self.metrics})
        fetch.append({"all_metrics": self.all_metrics})
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
        # best_thresh_results = metrics.get_best_metric_values(numpy_dict["all_metrics"], results)

        return results
    
    @doc_inherit
    def get_output_image(self):
        return self.norm_variance
    
    @doc_inherit
    def get_prediction(self):
        return self.prediction
        
    @doc_inherit
    def get_weights(self):
        return self.weights
