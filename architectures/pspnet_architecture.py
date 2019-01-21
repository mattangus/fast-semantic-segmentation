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


class PSPNetArchitecture(model.FastSegmentationModel):
    """PSPNet Architecture definition."""

    def __init__(self,
                is_training,
                model_arg_scope,
                num_classes,
                feature_extractor,
                classification_loss,
                dist_loss,
                filter_scale,
                use_aux_loss=True,
                main_loss_weight=1,
                aux_loss_weight=0,
                dist_loss_weight=1,
                add_summaries=True,
                scope=None,
                scale_pred=False,
                train_reduce=False):
        super(PSPNetArchitecture, self).__init__(num_classes=num_classes)
        self._is_training = is_training
        self._model_arg_scope = model_arg_scope
        self._num_classes = num_classes
        self._feature_extractor = feature_extractor
        self._classification_loss = classification_loss
        self._dist_loss = dist_loss
        self._filter_scale = filter_scale
        self._use_aux_loss = use_aux_loss
        self._main_loss_weight = main_loss_weight
        self._aux_loss_weight = aux_loss_weight
        self._dist_loss_weight = dist_loss_weight
        self._add_summaries = add_summaries
        self._scale_pred = scale_pred
        self._train_reduce = train_reduce

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
        if inputs.dtype is not tf.float32:
            raise ValueError('`preprocess` expects a tf.float32 tensor')
        with tf.name_scope('Preprocessor'):
            return self._feature_extractor.preprocess(inputs)

    def _extract_shared_features(self, preprocessed_inputs, scope):
        return self._feature_extractor.extract_features(
                        preprocessed_inputs, scope=scope)

    def predict(self, preprocessed_inputs, scope=None):
        """Build main inference pass"""
        with slim.arg_scope(self._model_arg_scope):
            # Feature extraction from arbitrary extractor
            _, backbone_logits, psp_aux_out = self._extract_shared_features(
                 preprocessed_inputs,
                 scope=self.shared_feature_extractor_scope)
            # Branch specific layers
            final_logits = self._pspnet_pspmodule(backbone_logits)
            
            # Class class_predictions
            with tf.variable_scope('Predictions'):
                # TODO: remove hack to not load predictions
                def pred_fun():
                    return slim.conv2d(final_logits, self._num_classes,
                                1, 1, activation_fn=None, normalizer_fn=None)
                if self._train_reduce:
                    with tf.variable_scope("dim_reduce"):
                        predictions = pred_fun()
                else:
                    predictions = pred_fun()
                
                unscaled_logits = predictions

                if self._train_reduce:
                    predictions_no_resize = final_logits
                else:
                    predictions_no_resize = predictions
                if not self._is_training: # evaluation
                    predictions = self._dynamic_interpolation(
                            predictions, z_factor=8.0)
            # Outputs with auxilarary loss for training
            prediction_dict = {
                self.main_class_predictions_key: predictions,
                self.final_logits_key: final_logits,
                self.unscaled_logits_key: unscaled_logits }
            # Aux loss as described in PSPNet paper
            if self._is_training and self._use_aux_loss:
                with tf.variable_scope('AuxPredictions'):
                    aux_preds= slim.conv2d(psp_aux_out,
                            self._num_classes, 1, 1,
                            activation_fn=None, normalizer_fn=None)
                prediction_dict[self.aux_predictions_key] = aux_preds
            
            return prediction_dict

    def _pspnet_pspmodule(self, input_features):
        """PSP Module """
        with tf.variable_scope('PSPModule'):
            input_n, input_h, input_w, input_c = input_features.get_shape().as_list()
            
            # full scale
            full_pool_in = slim.avg_pool2d(input_features,
                    [input_h, input_w], stride=[input_h, input_w])
            full_pool_conv = slim.conv2d(full_pool_in,
                    512, (1, 1),
                    stride=1, normalizer_fn=slim.batch_norm)
            full_pool = tf.image.resize_bilinear(full_pool_conv,
                    size=(input_h, input_w), align_corners=True)
            # 1/2 scale
            half_pool_in = slim.avg_pool2d(input_features,
                    [input_h/2, input_w/2], stride=[input_h/2, input_w/2])
            half_pool_conv = slim.conv2d(half_pool_in,
                    512, (1, 1),
                    stride=1, normalizer_fn=slim.batch_norm)
            half_pool = tf.image.resize_bilinear(half_pool_conv,
                    size=(input_h, input_w), align_corners=True)
            # 1/3 scale
            third_pool_in = slim.avg_pool2d(input_features,
                    [input_h/3, input_w/3], stride=[input_h/3, input_w/3])
            third_pool_conv = slim.conv2d(third_pool_in,
                    512, (1, 1),
                    stride=1, normalizer_fn=slim.batch_norm)
            third_pool = tf.image.resize_bilinear(third_pool_conv,
                    size=(input_h, input_w), align_corners=True)
            # 1/6 scale
            forth_pool_in = slim.avg_pool2d(input_features,
                    [input_h/6, input_w/6], stride=[input_h/6, input_w/6])
            forth_pool_conv = slim.conv2d(forth_pool_in,
                    512, (1, 1),
                    stride=1, normalizer_fn=slim.batch_norm)
            forth_pool = tf.image.resize_bilinear(forth_pool_conv,
                    size=(input_h, input_w), align_corners=True)
            # concat all
            branch_merge = tf.concat([input_features, full_pool,
                                     half_pool, third_pool, forth_pool],
                                     axis=-1)
            # if self._train_reduce:
            #     branch_merge = tf.stop_gradient(branch_merge)
            output = slim.conv2d(branch_merge,
                    512//self._filter_scale, (3, 3),
                    stride=1, normalizer_fn=slim.batch_norm)
            
            if self._train_reduce:
                #dimensionality reduction
                with tf.variable_scope("dim_reduce"):
                    output = slim.conv2d(output,
                            32, (3, 3),
                            stride=1, normalizer_fn=slim.batch_norm)
            return output

    def _dynamic_interpolation(self, features_to_upsample,
                               s_factor=1.0, z_factor=1.0):
        with tf.name_scope('Interp'):
            _, input_h, input_w, _ = features_to_upsample.get_shape().as_list()
            shrink_h = (input_h-1)*s_factor+1
            shrink_w = (input_w-1)*s_factor+1
            zoom_h = shrink_h + (shrink_h-1)*(z_factor-1)
            zoom_w = shrink_w + (shrink_w-1)*(z_factor-1)
            return tf.image.resize_bilinear(features_to_upsample,
                                            size=[int(zoom_h), int(zoom_w)],
                                            align_corners=True)

    def loss(self, prediction_dict, scope=None):
        losses_dict = {}
         # TODO: Make this an optional choice. For now only scale
        # down labels like in original paper
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

        if self._use_aux_loss and self._is_training:
            aux_preds = prediction_dict[self.aux_predictions_key]
            with tf.name_scope('AuxLoss'): # 1/8th labels
                if self._scale_pred:
                    aux_scaled_pred = _resize_logits_to_labels(aux_preds,
                        self._groundtruth_labels)
                else:
                    aux_scaled_pred = aux_preds
                aux_scaled_labels = main_scaled_labels
                first_aux_loss = self._classification_loss(aux_scaled_pred,
                                                        aux_scaled_labels)
                losses_dict[self.aux_loss_key] = (
                    self._aux_loss_weight * first_aux_loss)
        
        if self._dist_loss is not None and self._is_training:
            final_logits = prediction_dict[self.final_logits_key]
            with tf.name_scope('DistLoss'):
                dist_loss = self._dist_loss(final_logits, main_scaled_labels)
                losses_dict[self.dist_loss_key] = (
                    self._dist_loss_weight * dist_loss)

        return losses_dict

    def restore_map(self, checkpoint_path,
                    fine_tune_checkpoint_type='segmentation'):
        """Restore variables for checkpoints correctly"""
        if fine_tune_checkpoint_type not in [
                    'segmentation', 'resnet']:
            raise ValueError('Not supported fine_tune_checkpoint_type: {}'.format(
                fine_tune_checkpoint_type))
        if fine_tune_checkpoint_type == 'resnet':
            tf.logging.info('Fine-tuning from resnet checkpoints.')
            return self._feature_extractor.restore_from_classif_checkpoint_fn(
                self.shared_feature_extractor_scope)
        exclude_list = ['global_step']
        variables_to_restore = slim.get_variables_to_restore(
                                        exclude=exclude_list)
        #check the output layers
        reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
        var_to_shape_map = reader.get_variable_to_shape_map()
        variables_to_restore_new = []
        for v in variables_to_restore:
            name = v.name.replace(":0", "")
            if name in var_to_shape_map:
                v_shape = np.array(v.shape.as_list())
                save_shape = np.array(var_to_shape_map[name])
                if np.all(v_shape == save_shape):
                    variables_to_restore_new.append(v)
                else:
                    print("shape missmatch in:", v, ". skipping")
        return variables_to_restore_new


class PSPNetFeatureExtractor(object):
    """PSPNet ICNet Based Feature Extractor definition."""

    def __init__(self,
                 is_training,
                 features_stride,
                 batch_norm_trainable=False,
                 reuse_weights=None,
                 weight_decay=0.0):
        self._is_training = is_training
        self._features_stride = features_stride
        self._train_batch_norm = (batch_norm_trainable and is_training)
        self._reuse_weights = reuse_weights
        self._weight_decay = weight_decay

    @abstractmethod
    def preprocess(self, resized_inputs):
        pass

    def extract_features(self, preprocessed_inputs, scope=None):
        """Extracts half resolution features."""
        with tf.variable_scope(
                scope, values=[preprocessed_inputs], reuse=tf.AUTO_REUSE):
            return self._extract_features(preprocessed_inputs, scope)

    @abstractmethod
    def _extract_features(self, preprocessed_inputs, scope):
        pass

    def restore_from_classif_checkpoint_fn(self, scope_name):
        variables_to_restore = {}
        for variable in tf.global_variables():
            if variable.op.name.startswith(scope_name):
                var_name = variable.op.name.replace(scope_name + '/', '')
                variables_to_restore[var_name] = variable
        return variables_to_restore
