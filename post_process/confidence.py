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

class ConfidenceProcessor(pp.PostProcessor):

    def __init__(self, model, outputs_dict, num_classes,
                    annot, image, ignore_label, process_annot,
                    num_gpus, batch_size,
                    #class specific
                    epsilon):
        super().__init__("Dropout", model, outputs_dict, num_gpus)
        self.num_classes = num_classes - 1
        self.annot = annot
        self.image = image
        self._epsilon = epsilon
        self.ignore_label = ignore_label
        self._process_annot = process_annot
        self._batch_size = batch_size

        self.metric_gpu = "gpu:0"
        if self.num_gpus > 1:
            self.metric_gpu = "gpu:1"

    def _dot_fn(self, x):
        return tf.squeeze(tf.matmul(tf.expand_dims(x,-2), tf.expand_dims(x,-1)),-1)

    @doc_inherit
    def post_process_ops(self):
        main_pred = self.outputs_dict[self.model.main_class_predictions_key]
        unscaled_logits = self.outputs_dict[self.model.unscaled_logits_key]
        pred_shape = main_pred.shape.as_list()

        self.weights = tf.to_float(get_valid(self.annot, self.ignore_label))
        self.annot_before = self.annot
        self.annot, self.num_classes = self._process_annot(self.annot, main_pred, self.num_classes)

        static_shape = unscaled_logits.shape.as_list()
        unscaled_logits.set_shape([self._batch_size] + static_shape[1:])

        conf_logits = tf.image.resize_bilinear(unscaled_logits[...,-1:], pred_shape[1:3])

        self.uncertainty = 1. - tf.nn.sigmoid(conf_logits)

        self.metrics, self.update = metrics.get_metric_ops(self.annot, self.uncertainty, self.weights)

    @doc_inherit
    def get_init_feed(self):
        return {}

    @doc_inherit
    def get_preprocessed(self):
        return None

    @doc_inherit
    def get_vars_noload(self):
        return []

    @doc_inherit
    def get_fetch_dict(self):
        fetch = [{"update": self.update}, {"metrics": self.metrics}]
        return fetch

    @doc_inherit
    def get_feed_dict(self):
        return {}

    @doc_inherit
    def post_process(self, numpy_dict):
        results = metrics.get_metric_values(numpy_dict["metrics"])

        return results