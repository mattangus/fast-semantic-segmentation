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

    def _dot_fn(self, x):
        return tf.squeeze(tf.matmul(tf.expand_dims(x,-2), tf.expand_dims(x,-1)),-1)

    @doc_inherit
    def post_process_ops(self):
        main_pred = self.outputs_dict[self.model.main_class_predictions_key]
        unscaled_logits = self.outputs_dict[self.model.unscaled_logits_key]
        pred_shape = main_pred.shape.as_list()

        weights = tf.to_float(get_valid(self.annot, self.ignore_label))
        self.annot, self.num_classes = self._process_annot(self.annot, main_pred, self.num_classes)
        
        #m_k = m_{k-1} + (x_k - m_{k-1})/k
        #v_k = v_{k-1} + (x_k - m_{k-1})(x_k - m_k)
        static_shape = unscaled_logits.shape.as_list()

        stacked_logits = tf.stack(tf.split(unscaled_logits, self._num_runs))

        mean, var = tf.nn.moments(stacked_logits, [0])

        self.mean_logits = (tf.image.resize_bilinear(mean, pred_shape[1:3]))
        self.all_variance = (tf.image.resize_bilinear(var, pred_shape[1:3]))

        #weights = tf.expand_dims(weights, -1)
        # m_k = metrics_impl.metric_variable(static_shape, tf.float32, name="streaming_mean")
        # v_k = metrics_impl.metric_variable(static_shape, tf.float32, name="streaming_variance")
        # left = metrics_impl.metric_variable(static_shape, tf.float32, name="scratch")
        # k = metrics_impl.metric_variable((), tf.float32, name="k")
        
        # update_left = tf.assign(left, unscaled_logits - m_k)
        # update_k = tf.assign_add(k, 1)
        # with tf.control_dependencies([update_left, update_k]):
        #     update_mk = tf.assign_add(m_k, _safe_div(left, k))
        # with tf.control_dependencies([update_mk]):        
        #     update_vk = tf.assign(v_k, v_k + left*(unscaled_logits - m_k))

        # self.update_op = tf.group([update_left, update_mk, update_vk, update_k])

        # self.reset_op = tf.group([tf.assign(v, tf.zeros_like(v)) for v in [k, m_k, v_k]])

        # variance = v_k / k

        # self.all_variance = (tf.image.resize_bilinear(variance, pred_shape[1:3]))
        
        # self.mean_logits = (tf.image.resize_bilinear(m_k, pred_shape[1:3]))

        self.prediction = tf.cast(tf.expand_dims(tf.argmax(self.mean_logits, -1),-1), tf.int32)

        xs = [list(range(s)) for s in [self._batch_size] + pred_shape[1:-1]]
        grid = np.meshgrid(*xs, indexing="ij")
        grid = list(map(lambda x: np.expand_dims(x,-1), grid))
        grid_pl = list(map(lambda x: tf.placeholder(tf.int32, shape=x.shape), grid))
        self.idx = tf.concat(grid_pl + [self.prediction], -1)

        self.feed = {}
        for pl, v in zip(grid_pl, grid):
            self.feed[pl] = v

        self.interp_variance = tf.expand_dims(tf.gather_nd(self.all_variance, self.idx),-1, name="interp_variance")

        self.metrics, self.update = metrics.get_metric_ops(self.annot, self.interp_variance, weights)

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
        fetch.append({
            "all_var": self.all_variance,
            "pred": self.prediction,
            "res": self.interp_variance,})
        fetch.append({"update": self.update})
        fetch.append({"metrics": self.metrics})
        # fetch.append({"reset": self.reset_op})
        return fetch
    
    @doc_inherit
    def get_feed_dict(self):
        return self.feed
    
    @doc_inherit
    def post_process(self, numpy_dict):
        import matplotlib.pyplot as plt
        plt.imshow(numpy_dict["res"][0,...,0])
        plt.show()
        import pdb; pdb.set_trace()
        results = metrics.get_metric_values(numpy_dict["metrics"])

        return results