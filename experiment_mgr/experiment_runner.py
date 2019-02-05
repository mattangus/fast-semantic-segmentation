r"""Run inference on an image or group of images."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import timeit
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import sys
import logging
import traceback

from builders import model_builder, dataset_builder
from post_process.mahalanobis import MahalProcessor
from post_process.max_softmax import MaxSoftmaxProcessor
from protos.config_reader import read_config
from libs.exporter import deploy_segmentation_inference_graph

def ood_annot(annot, prediction, num_classes):
    annot = tf.to_float(annot >= num_classes)
    return annot, 2

def error_annot(annot, prediction, num_classes):
    not_correct = tf.to_float(tf.not_equal(annot, tf.to_float(prediction)))
    return not_correct, 2

annot_dict = {
    "ood": ood_annot,
    "error": error_annot,
}

processor_dict = {
    "Mahal": MahalProcessor,
    "MaxSoftmax": MaxSoftmaxProcessor,
}

def run_inference_graph(model, trained_checkpoint_prefix,
                        dataset, num_images, ignore_label, pad_to_shape,
                        num_classes, processor_type, annot_type, **kwargs):
    batch = 1

    dataset = dataset.batch(batch, drop_remainder=True)
    dataset = dataset.apply(tf.data.experimental.ignore_errors())
    data_iter = dataset.make_one_shot_iterator()
    input_dict = data_iter.get_next()

    input_tensor = input_dict[dataset_builder._IMAGE_FIELD]
    annot_tensor = input_dict[dataset_builder._LABEL_FIELD]
    input_name = input_dict[dataset_builder._IMAGE_NAME_FIELD]

    annot_pl = tf.placeholder(tf.float32, annot_tensor.get_shape().as_list(), name="annot_pl")
    outputs, placeholder_tensor = deploy_segmentation_inference_graph(
        model=model,
        input_shape=input_tensor.shape.as_list(),
        #input=input_tensor,
        pad_to_shape=pad_to_shape,
        input_type=tf.float32)

    process_annot = annot_dict[annot_type]
    processor_class = processor_dict[processor_type]

    processor = processor_class(model, outputs, num_classes,
                            annot_pl, placeholder_tensor, ignore_label,
                            process_annot, **kwargs)

    # if processor_type == "MaxSoftmax":
    #     processor = MaxSoftmaxProcessor(model, outputs, num_classes,
    #                         annot_pl, placeholder_tensor,
    #                         FLAGS.epsilon, FLAGS.t_value, ignore_label,
    #                         ood_annot)
    # elif processor_type == "Mahal":
    #     processor = MahalProcessor(model, outputs, num_classes, annot_pl,
    #                         placeholder_tensor, eval_dir, FLAGS.epsilon,
    #                         FLAGS.global_cov, FLAGS.global_mean, ignore_label,
    #                         ood_annot)
    # else:
    #     raise ValueError(str(processor_type) + " is an unknown processor")

    processor.post_process_ops()

    preprocess_input = processor.get_preprocessed()

    input_fetch = [input_name, input_tensor, annot_tensor]

    fetch = processor.get_fetch_dict()
    feed = processor.get_feed_dict()

    num_step = num_images // batch
    print("running for", num_step, "steps")

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        init_feed = processor.get_init_feed()
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()],init_feed)

        vars_noload = set(processor.get_vars_noload())
        vars_toload = [v for v in tf.global_variables() if v not in vars_noload]
        saver = tf.train.Saver(vars_toload)
        saver.restore(sess, trained_checkpoint_prefix)

        for idx in range(num_step):

            start_time = timeit.default_timer()

            inputs = sess.run(input_fetch)

            annot_raw = inputs[2]
            img_raw = inputs[1]
            image_path = inputs[0][0].decode("utf-8")

            if preprocess_input is not None:
                processed_input = sess.run(preprocess_input, feed_dict={placeholder_tensor: img_raw, annot_pl: annot_raw})
            else:
                processed_input = img_raw

            feed_dict = {
                placeholder_tensor: processed_input,
                annot_pl: annot_raw
            }

            feed_dict.update(feed)

            res = {}
            for f in fetch:
                res.update(sess.run(f, feed_dict))

            result = processor.post_process(res)

            elapsed = timeit.default_timer() - start_time
            end = "\r"
            if idx % 50 == 0:
                #every now and then do regular print
                end = "\n"
            print('{0:.4f} iter: {1}, {2}'.format(elapsed, idx+1, result), end=end) 
        print('{0:.4f} iter: {1}, {2}'.format(elapsed, idx+1, result))
        return result


def run_experiment(gpus, print_buffer, model_config, data_config,
                    trained_checkpoint, pad_to_shape,
                    processor_type, annot_type, **kwargs):
    had_error = None
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
        sys.stdout = print_buffer
        sys.stderr = print_buffer

        pipeline_config = read_config(model_config, data_config)

        if pad_to_shape is not None and isinstance(pad_to_shape, str) :
            pad_to_shape = [
                int(dim) if dim != '-1' else None
                    for dim in pad_to_shape.split(',')]

        input_reader = pipeline_config.input_reader
        input_reader.shuffle = True
        ignore_label = input_reader.ignore_label

        num_classes, segmentation_model = model_builder.build(
            pipeline_config.model, is_training=False, ignore_label=ignore_label)
        dataset = dataset_builder.build(input_reader, 1)

        result = run_inference_graph(segmentation_model, trained_checkpoint, dataset,
                            input_reader.num_examples, ignore_label, pad_to_shape,
                            num_classes, processor_type, annot_type, **kwargs)
        had_error = False
    except Exception:
        print(traceback.format_exc())
        had_error = True
        result = None
    
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    tf.reset_default_graph()

    return print_buffer, result, had_error

