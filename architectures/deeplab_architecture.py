r"""PSPNet Semantic Segmentation architecture.

As described in http://arxiv.org/abs/1612.01105.

  Pyramid Scene Parsing Network
  Hengshuang Zhao, Jianping Shi, Xiaojuan Qi, Xiaoyong Shen, Jiaya Jia

This is a baseline architecture that was implemented for the purposes
of validating the other segmentation models which are more efficient.

Please note that although this network is accurate, it is VERY
slow and memory intensive. It should not be used under the assumption that
it will result in similar performance as the other models in this project.
"""
from abc import abstractmethod
from functools import partial
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import numpy as np

from . import base_model as model

slim = tf.contrib.slim

from deeplab import common
from deeplab import model as dl_model

class DeeplabArchitecture(model.SegmentationModel):
    """Deeplab Architecture definition."""

    def __init__(self,
                is_training,
                num_classes,
                classification_loss,
                feature_extractor,
                use_aux_loss=True,
                main_loss_weight=1,
                aux_loss_weight=0,
                add_summaries=True,
                scope=None,
                scale_pred=False,
                train_reduce=False):
        super(DeeplabArchitecture, self).__init__(num_classes=num_classes)
        self._is_training = is_training
        self._num_classes = num_classes
        self._classification_loss = classification_loss
        self._use_aux_loss = use_aux_loss
        self._main_loss_weight = main_loss_weight
        self._feature_extractor = feature_extractor
        self._aux_loss_weight = aux_loss_weight
        self._add_summaries = add_summaries
        self._scale_pred = scale_pred
        self._train_reduce = train_reduce
        self.model_options = common.ModelOptions(
            outputs_to_num_classes={common.OUTPUT_TYPE: num_classes},
            crop_size=None,
            atrous_rates=[12, 24, 36],
            output_stride=8)
        self.model_options = self.model_options._replace(model_variant=feature_extractor)

    @property
    def main_class_predictions_key(self):
        return 'class_predictions'

    @property
    def unscaled_logits_key(self):
        return 'unscaled_logits'
    
    @property
    def final_logits_key(self):
        return 'final_logits'

    @property
    def aux_predictions_key(self):
        return 'aux_predictions'

    @property
    def main_loss_key(self):
        return 'loss'

    @property
    def aux_loss_key(self):
        return 'aux_loss'
    
    @property
    def dist_loss_key(self):
        return 'dist_loss'

    def preprocess(self, inputs):
        return inputs

    def predict(self, preprocessed_inputs, scope=None):
        crop_size = tuple(preprocessed_inputs.shape.as_list()[1:3])
        self.model_options = self.model_options._replace(crop_size=crop_size)
        
        tmp_options = self.model_options
        if self._train_reduce:
            #HACK: Set output depth to 32 for dim redice then add conv to
            tmp_options = tmp_options._replace(outputs_to_num_classes={common.OUTPUT_TYPE: 32})
            
        predictions, logits = dl_model.predict_labels(preprocessed_inputs, tmp_options,
                                         image_pyramid=None)        
        final_logits = logits[common.OUTPUT_TYPE + "_unscaled"]
        def pred_fun():
            return slim.conv2d(logits[common.OUTPUT_TYPE + "_unscaled"], self._num_classes,
                        1, 1, normalizer_fn=None)
        if self._train_reduce:
            with tf.variable_scope("dim_reduce"):
                unscaled_logits = pred_fun()
            main_pred = dl_model._resize_bilinear(unscaled_logits,
                                tf.shape(preprocessed_inputs)[1:3],
                                unscaled_logits.dtype)
        else:
            unscaled_logits = final_logits
            main_pred =  logits[common.OUTPUT_TYPE]

        prediction_dict = {self.main_class_predictions_key: main_pred,
                           self.final_logits_key: final_logits,
                           self.unscaled_logits_key: unscaled_logits}
        return prediction_dict

    def loss(self, prediction_dict, scope=None):
        losses_dict = {}
        def _resize_labels_to_logits(labels, logits):
            logits_shape = logits.get_shape().as_list()
            scaled_labels = tf.image.resize_nearest_neighbor(
                labels, logits_shape[1:3], align_corners=True)
            return scaled_labels
        def _resize_logits_to_labels(logits, labels, s_factor=1):
            labels_h, labels_w = labels.get_shape().as_list()[1:3]
            new_logits_size = (labels_h//s_factor, labels_w//s_factor)
            scaled_logits = tf.image.resize_bilinear(
                logits, new_logits_size, align_corners=True)
            return scaled_logits

        main_preds = prediction_dict[self.main_class_predictions_key]
        with tf.name_scope('SegmentationLoss'): # 1/8th labels
            if self._scale_pred:
                main_scaled_pred = _resize_logits_to_labels(main_preds,
                    self._groundtruth_labels)
                main_scaled_labels = self._groundtruth_labels
            else:
                main_scaled_labels = _resize_labels_to_logits(
                    self._groundtruth_labels, main_preds)
                main_scaled_pred = main_preds
            
            main_scaled_pred = tf.identity(main_scaled_pred, name="ScaledPreds")
            main_scaled_labels = tf.identity(main_scaled_labels, name="ScaledLabels")

            main_loss = self._classification_loss(main_scaled_pred,
                                            main_scaled_labels)
            losses_dict[self.main_loss_key] = (
                self._main_loss_weight * main_loss)

        return losses_dict

    def restore_map(self, checkpoint_path,
                    fine_tune_checkpoint_type='segmentation'):
        """Restore variables for checkpoints correctly"""
        if fine_tune_checkpoint_type not in [
                    'segmentation', 'resnet']:
            raise ValueError('Not supported fine_tune_checkpoint_type: {}'.format(
                fine_tune_checkpoint_type))
        # import pdb; pdb.set_trace()
        # if fine_tune_checkpoint_type == 'resnet':
        #     tf.logging.info('Fine-tuning from resnet checkpoints.')
        #     return self._feature_extractor.restore_from_classif_checkpoint_fn(
        #         self.shared_feature_extractor_scope)
        exclude_list = ['global_step']
        variables_to_restore = slim.get_variables_to_restore(
                                        exclude=exclude_list)
        #check the output layers
        reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
        var_to_shape_map = reader.get_variable_to_shape_map()
        variables_to_restore_new = []
        tf.logging.info("validating checkpoint")
        for v in variables_to_restore:
            name = v.name.replace(":0", "")
            if name in var_to_shape_map:
                v_shape = np.array(v.shape.as_list())
                save_shape = np.array(var_to_shape_map[name])
                if np.all(v_shape == save_shape):
                    variables_to_restore_new.append(v)
                else:
                    print("shape missmatch in:", v, ". skipping")
        # import pdb; pdb.set_trace()
        return variables_to_restore_new
