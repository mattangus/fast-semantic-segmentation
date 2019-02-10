from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import six
import tensorflow as tf
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np

from builders import dataset_builder
from builders import preprocessor_builder as preprocessor

from libs import sliding_window
from libs import metrics

slim = tf.contrib.slim

prefetch_queue = slim.prefetch_queue


def create_evaluation_input(create_input_dict_fn,
                            input_height,
                            input_width,
                            cropped_eval=False):
    dataset = create_input_dict_fn(num_epoch=0)
    def to_float(input_dict):
        input_dict[dataset_builder._IMAGE_FIELD] = tf.to_float(input_dict[dataset_builder._IMAGE_FIELD])
        input_dict[dataset_builder._LABEL_FIELD] = tf.to_float(input_dict[dataset_builder._LABEL_FIELD])
        return input_dict
    if cropped_eval:
        # We evaluate on a random cropped of the validation set.
        # cropper_fn = functools.partial(preprocessor.random_crop,
        #                crop_height=input_height,
        #                crop_width=input_width)
        # output_dict = preprocessor.preprocess_runner(
        #         input_dict, func_list=[cropper_fn])
        # processed_labels = tf.to_float(
        #     output_dict[dataset_builder._LABEL_FIELD])
        # processed_images = tf.to_float(input_dict[dataset_builder._IMAGE_FIELD])
        # #sliding_window.extract_patches(input_dict[dataset_builder._IMAGE_FIELD], input_width, input_height)
        # processed_labels = tf.to_float(input_dict[dataset_builder._LABEL_FIELD])
        #sliding_window.extract_patches(input_dict[dataset_builder._LABEL_FIELD], input_width, input_height)
        dataset = dataset.map(to_float)
    else:
        # Here we only pad input image, then we shrink back the prediction
        padding_fn = functools.partial(preprocessor.pad_to_specific_size,
                        height_to_set=input_height,
                        width_to_set=input_width)
        def pre_process(input_dict):
            output_dict = preprocessor.preprocess_runner(
                    input_dict, skip_labels=True, func_list=[padding_fn])
            input_dict[dataset_builder._IMAGE_FIELD] = output_dict[dataset_builder._IMAGE_FIELD]
            return input_dict
        
        dataset = dataset.map(pre_process)
        dataset = dataset.map(to_float)
    return dataset


def create_predictions_and_labels(model, create_input_dict_fn,
                                 input_height, input_width, cropped_eval,
                                 eval_dir=None, num_extra_class=0):
    dataset = create_evaluation_input(
        create_input_dict_fn, input_height, input_width, cropped_eval)
    
    # Setup a queue for feeding to slim evaluation helpers
    data_iterator = dataset.make_one_shot_iterator()
    input_dict = data_iterator.get_next()
    eval_images = input_dict[dataset_builder._IMAGE_FIELD]
    eval_labels = input_dict[dataset_builder._LABEL_FIELD]

    eval_labels = tf.expand_dims(eval_labels, 0)
    eval_images = tf.stack([eval_images, tf.image.flip_left_right(eval_images)])
    #eval_images = tf.expand_dims(eval_images, 0)

    if cropped_eval:
        eval_images_orig = eval_images
        c = int(eval_images.get_shape()[-1])
        eval_images = sliding_window.extract_patches(eval_images, input_height, input_width)
        patches_shape = tf.shape(eval_images)
        eval_images = tf.reshape(eval_images, [tf.reduce_prod(patches_shape[0:3]), input_height, input_width, c])

    # Main predictions
    mean_subtracted_inputs = model.preprocess(eval_images)
    model.provide_groundtruth(eval_labels)
    output_dict = model.predict(mean_subtracted_inputs)

    # Awkward fix from preprocessing step - we resize back down to label shape
    if not cropped_eval:
        eval_labels_shape = eval_labels.get_shape().as_list()
        padded_predictions = output_dict[model.main_class_predictions_key]
        padded_predictions = tf.image.resize_bilinear(padded_predictions,
            size=eval_labels_shape[1:3],
            align_corners=True)
        output_dict[model.main_class_predictions_key] = padded_predictions

    model_scores = output_dict[model.main_class_predictions_key]
    model_scores = tf.split(model_scores, 2, 0)
    model_scores[1] = tf.image.flip_left_right(model_scores[1])
    model_scores = tf.reduce_mean(tf.stack(model_scores), 0)
    output_dict[model.main_class_predictions_key] = model_scores
    if cropped_eval:
        target = tf.zeros(eval_labels.get_shape().as_list()[:-1] + model_scores.get_shape().as_list()[-1:])
        print("Merging patches. Could take a while.")
        model_scores = sliding_window.merge_patches(target, model_scores, input_height, input_width)
        print("Done merging!")
        output_dict[model.main_class_predictions_key] = model_scores
    if num_extra_class != 0:
        model_scores = model_scores[...,:-num_extra_class]
    eval_predictions = tf.argmax(model_scores, 3)
    eval_predictions = tf.expand_dims(eval_predictions, -1)



    # Output graph def for pruning
    if eval_dir is not None:
        graph_def = tf.get_default_graph().as_graph_def()
        pred_graph_def_path = os.path.join(eval_dir, "eval_graph.pbtxt")
        f = tf.gfile.FastGFile(pred_graph_def_path, "w")
        f.write(str(graph_def))
    # Validation loss to fight overfitting
    validation_losses = model.loss(output_dict)
    eval_total_loss =  sum(validation_losses.values())
    # Argmax final outputs to feed to a metric function


    return eval_predictions, eval_labels, eval_images, eval_total_loss


def eval_segmentation_model_once(checkpoint_path,
                                 create_model_fn,
                                 create_input_fn,
                                 input_dimensions,
                                 eval_config,
                                 input_reader,
                                 eval_dir,
                                 num_extra_class=0,
                                 cropped_evaluation=False,
                                 image_summaries=False,
                                 verbose=False):
    return eval_segmentation_model(
        create_model_fn,
        create_input_fn,
        input_dimensions,
        eval_config,
        num_extra_class=num_extra_class,
        input_reader=input_reader,
        train_dir=None,
        eval_dir=eval_dir,
        cropped_evaluation=cropped_evaluation,
        evaluate_single_checkpoint=checkpoint_path,
        image_summaries=image_summaries,
        verbose=verbose)

def confusion_matrix_op(labels,
                        predictions,
                        num_classes,
                        dtype=tf.float64,
                        name=None,
                        weights=None):
    cur_conf = tf.confusion_matrix(labels, predictions, num_classes, dtype, name, weights)
    avg_conf = tf.get_variable("confusion_mat", [num_classes, num_classes], dtype=dtype)
    # constant for same dtype
    one = tf.constant(1.0, dtype=dtype)
    t = tf.get_variable("conf_t", dtype=dtype, initializer=one)
    up_1 = tf.assign_add(avg_conf, (cur_conf - avg_conf)/t)
    up_2 = tf.assign_add(t, one)
    update_op = [up_1, up_2]
    return (avg_conf, t), update_op

def eval_segmentation_model(create_model_fn,
                            create_input_fn,
                            input_dimensions,
                            eval_config,
                            input_reader,
                            train_dir,
                            eval_dir,
                            num_extra_class=0,
                            cropped_evaluation=False,
                            evaluate_single_checkpoint=None,
                            image_summaries=False,
                            verbose=False):
    num_classes, segmentation_model = create_model_fn()

    input_height, input_width = input_dimensions
    (predictions_for_eval, labels_for_eval, inputs_summary,
      validation_loss_summary) = create_predictions_and_labels(
                model=segmentation_model,
                create_input_dict_fn=create_input_fn,
                input_height=input_height,
                input_width=input_width,
                cropped_eval=cropped_evaluation,
                eval_dir=eval_dir,
                num_extra_class=num_extra_class)
    variables_to_restore = tf.global_variables()
    global_step = tf.train.get_or_create_global_step()
    variables_to_restore.append(global_step)

    # Prepare inputs to metric calculation steps
    flattened_predictions = tf.reshape(predictions_for_eval, shape=[-1])
    flattened_labels = tf.reshape(labels_for_eval, shape=[-1])

    ne = [tf.not_equal(flattened_labels, il) for il in input_reader.ignore_label]
    neg_validity_mask = ne.pop(0)
    for v in ne:
        neg_validity_mask = tf.logical_and(neg_validity_mask, v)
    #validity_mask = tf.logical_not(neg_validity_mask) #tf.equal(flattened_labels, ignore_label)
    #neg_validity_mask = tf.not_equal(flattened_labels, ignore_label)
    eval_labels = tf.where(neg_validity_mask, flattened_labels,
                        tf.zeros_like(flattened_labels))
    # Calculate metrics from predictions
    metric_map = {}
    predictions_tag='EvalMetrics/mIoU'
    conf_tag='EvalMetrics/confusionMatrix'
    value_op, conf_value, update_op = metrics.mean_iou(
                        flattened_predictions, eval_labels, num_classes,
                        weights=tf.to_float(neg_validity_mask))
    # conf_value, conf_update =  metrics._streaming_confusion_matrix(eval_labels,
    #                     flattened_predictions, num_classes,
    #                     weights=tf.to_float(neg_validity_mask))
    # (conf_value, t_value), conf_update = confusion_matrix_op(flattened_predictions, 
    #                                             eval_labels, num_classes,
    #                                             weights=tf.to_float(neg_validity_mask))
    # TODO: Extend the metrics tuple if needed in the future
    metric_map[predictions_tag] = (value_op, update_op)
    metric_map[conf_tag] = (conf_value, update_op)
    # Print updates if verbosity is requested
    if verbose:
        update_op = tf.Print(update_op, [value_op], predictions_tag)
        #conf_update = tf.Print(conf_update, [conf_value], conf_tag)
    value_op = [value_op, conf_value]
    # update_op = [update_op, conf_update]
    metrics_to_values, metrics_to_updates = (
        tf.contrib.metrics.aggregate_metric_map(metric_map))
    for metric_name, metric_value in six.iteritems(metrics_to_values):
        if "mIoU" in metric_name:
            tf.summary.scalar(metric_name,  metric_value)
    eval_op = list(metrics_to_updates.values())

    # Summaries for Tensorboard
    if validation_loss_summary is not None:
        tf.summary.scalar("Losses/EvalValidationLoss",
            validation_loss_summary)
    # Image summaries if requested
    if image_summaries:
        pixel_scaling = max(1, 255 // num_classes)
        tf.summary.image(
            'InputImage', inputs_summary, family='EvalImages')
        groundtruth_viz = tf.cast(labels_for_eval*pixel_scaling, tf.uint8)
        tf.summary.image(
            'GroundtruthImage', groundtruth_viz, family='EvalImages')
        predictions_viz = tf.cast(predictions_for_eval*pixel_scaling, tf.uint8)
        tf.summary.image(
            'PredictionImage', predictions_viz, family='EvalImages')
    summary_op = tf.summary.merge_all()

    tf.logging.info('Evaluating over %d samples...',
                    input_reader.num_examples)

    total_eval_examples = input_reader.num_examples
    if evaluate_single_checkpoint:
        curr_checkpoint = evaluate_single_checkpoint 
        metric_results = slim.evaluation.evaluate_once(
                            master='',
                            checkpoint_path=curr_checkpoint,
                            logdir=eval_dir,
                            num_evals=total_eval_examples,
                            eval_op=eval_op,
                            final_op=value_op,
                            summary_op=summary_op,
                            variables_to_restore=variables_to_restore)
        tf.logging.info('Evaluation of `{}` over. Eval values: {}'.format(
                    curr_checkpoint, metric_results))
    else:
        metric_results = slim.evaluation.evaluation_loop(
                            master='',
                            checkpoint_dir=train_dir,
                            logdir=eval_dir,
                            num_evals=total_eval_examples,
                            eval_op=eval_op,
                            final_op=value_op,
                            summary_op=summary_op,
                            #timeout=0,
                            variables_to_restore=variables_to_restore)
        
        tf.logging.info('Evaluation over. Eval values: {}'.format(
                        metric_results))
    cm = metric_results[1]
    cm = 100 * (cm / np.sum(cm, axis=0))
    plt.figure(figsize=[20.48,10.24])
    sb.heatmap(cm, annot=True)
    plt.savefig(eval_dir + "/confusion_matrix.png")
    plt.cla() #clear memory since no display happens
    plt.clf()

    return metric_results
