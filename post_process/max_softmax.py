import tensorflow as tf
import os
import numpy as np

from . import post_processor as pp
from third_party.doc_inherit import doc_inherit
from helpers import get_valid
from . import validation_metrics as metrics

class MaxSoftmaxProcessor(pp.PostProcessor):
    
    def __init__(self, model, outputs_dict, num_classes,
                    annot, image, path, ignore_label, process_annot,
                    num_gpus, batch_size,
                    #class specific
                    epsilon, t_value):
        super().__init__("MaxSoftmax", model, outputs_dict, num_gpus)
        self.num_classes = num_classes
        self.annot = annot
        self.image = image
        self.path = path
        self.epsilon = epsilon
        self.t_value = t_value
        self.ignore_label = ignore_label
        self._process_annot = process_annot
        self.batch_size = batch_size

        self.pre_process_gpu = "gpu:0"
        if num_gpus > 1:
            self.pre_process_gpu = "gpu:1"

    @doc_inherit
    def post_process_ops(self):
        main_pred = self.outputs_dict[self.model.main_class_predictions_key]
        unscaled_logits = self.outputs_dict[self.model.unscaled_logits_key]
        pred_shape = main_pred.shape.as_list()
        unscaled_logits.set_shape([self.batch_size] + unscaled_logits.shape.as_list()[1:])

        self.weights = tf.to_float(get_valid(self.annot, self.ignore_label))
        self.annot, self.num_classes = self._process_annot(self.annot, main_pred, self.num_classes)

        self.interp_logits = tf.image.resize_bilinear(unscaled_logits, pred_shape[1:3])
        self.prediction = 1.0 - tf.reduce_max(tf.nn.softmax(self.interp_logits/self.t_value), -1, keepdims=True)
        self.metrics, self.update = metrics.get_metric_ops(self.annot, self.prediction, self.weights)

    @doc_inherit
    def get_init_feed(self):
        return {}
        
    @doc_inherit
    def get_preprocessed(self):
        with tf.device(self.pre_process_gpu):
            if self.epsilon > 0.0:
                self.grads = tf.gradients(self.prediction, self.image)
                return self.image - tf.squeeze(self.epsilon*tf.sign(self.grads), 1)
            return None

    @doc_inherit
    def get_vars_noload(self):
        return []

    @doc_inherit
    def get_fetch_dict(self):
        return [{"update": self.update}, {"metrics": self.metrics}]
    
    @doc_inherit
    def get_feed_dict(self):
        return {}
    
    @doc_inherit
    def post_process(self, numpy_dict):
        results = metrics.get_metric_values(numpy_dict["metrics"])

        return results
    
    @doc_inherit
    def get_output_image(self):
        return self.prediction
    
    @doc_inherit
    def get_prediction(self):
        return self.outputs_dict[self.model.main_class_predictions_key]
    
    @doc_inherit
    def get_weights(self):
        return self.weights