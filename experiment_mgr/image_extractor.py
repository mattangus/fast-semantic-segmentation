r"""Run inference on an image or group of images."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import timeit
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import sys
import logging
import traceback
from io import StringIO

from builders import model_builder, dataset_builder
from post_process.mahalanobis import MahalProcessor
from post_process.max_softmax import MaxSoftmaxProcessor
from post_process.droput import DropoutProcessor
from post_process.confidence import ConfidenceProcessor
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
    "Dropout": DropoutProcessor,
    "Confidence": ConfidenceProcessor,
}

def run_inference_graph(model, trained_checkpoint_prefix,
                        dataset, num_images, ignore_label, pad_to_shape,
                        num_classes, processor_type, annot_type, num_gpu, **kwargs):
    batch = 1

    dataset = dataset.batch(batch, drop_remainder=True)
    dataset = dataset.apply(tf.data.experimental.ignore_errors())
    data_iter = dataset.make_one_shot_iterator()
    input_dict = data_iter.get_next()

    input_tensor = input_dict[dataset_builder._IMAGE_FIELD]
    annot_tensor = input_dict[dataset_builder._LABEL_FIELD]
    input_name = input_dict[dataset_builder._IMAGE_NAME_FIELD]

    input_shape = [None] + input_tensor.shape.as_list()[1:]

    name_pl = tf.placeholder(tf.string, input_name.shape.as_list(), name="name_pl")
    annot_pl = tf.placeholder(tf.float32, annot_tensor.shape.as_list(), name="annot_pl")
    outputs, placeholder_tensor = deploy_segmentation_inference_graph(
        model=model,
        input_shape=input_shape,
        #input=input_tensor,
        pad_to_shape=pad_to_shape,
        input_type=tf.float32)

    process_annot = annot_dict[annot_type]
    processor_class = processor_dict[processor_type]

    processor = processor_class(model, outputs, num_classes,
                            annot_pl, placeholder_tensor, name_pl, ignore_label,
                            process_annot, num_gpu, batch, **kwargs)

    processor.post_process_ops()

    preprocess_input = processor.get_preprocessed()

    input_fetch = [name_pl, input_tensor, annot_tensor]

    fetch = processor.get_fetch_dict()
    feed = processor.get_feed_dict()

    num_step = num_images // batch

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction=1.
    run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
    #config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        init_feed = processor.get_init_feed()
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()], init_feed)

        vars_noload = set(processor.get_vars_noload())
        vars_toload = [v for v in tf.global_variables() if v not in vars_noload]
        saver = tf.train.Saver(vars_toload)
        saver.restore(sess, trained_checkpoint_prefix)

        print("finalizing graph")
        sess.graph.finalize()

        #one sun image is bad
        num_step -= 1

        print("running for", num_step, "steps")
        for idx in range(num_step):

            start_time = timeit.default_timer()

            inputs = sess.run(input_fetch)

            annot_raw = inputs[2]
            img_raw = inputs[1]
            image_path = inputs[0]

            if preprocess_input is not None:
                processed_input = sess.run(preprocess_input, feed_dict={placeholder_tensor: img_raw, annot_pl: annot_raw, name_pl: image_path})
            else:
                processed_input = img_raw

            feed_dict = {
                placeholder_tensor: processed_input,
                annot_pl: annot_raw,
                name_pl: image_path
            }

            feed_dict.update(feed)

            output_image = sess.run(fetch, feed_dict, options=run_options)

            import pdb; pdb.set_trace()

            cv2.imshow("image", img_raw[0])
            cv2.imshow("uncertainty", output_image[0,...,0])
            
            print(image_path[0].decode())

            while True:
                key = cv2.waitKey()
                if key == 27: #escape
                    break
                elif key == 32: #space
                    break
                elif key == 115: #s
                    break


            


def extract_images(gpus, model_config, data_config,
                    trained_checkpoint, pad_to_shape,
                    processor_type, annot_type, is_debug, **kwargs):

    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    pipeline_config = read_config(model_config, data_config)

    if pad_to_shape is not None and isinstance(pad_to_shape, str) :
        pad_to_shape = [
            int(dim) if dim != '-1' else None
                for dim in pad_to_shape.split(',')]

    input_reader = pipeline_config.input_reader
    input_reader.shuffle = False
    ignore_label = input_reader.ignore_label

    num_classes, segmentation_model = model_builder.build(
        pipeline_config.model, is_training=False, ignore_label=ignore_label)
    with tf.device("cpu:0"):
        dataset = dataset_builder.build(input_reader, 1)

    num_gpu = len(gpus.split(","))

    num_examples = sum([r.num_examples for r in input_reader.tf_record_input_reader])

    run_inference_graph(segmentation_model, trained_checkpoint, dataset,
                        num_examples, ignore_label, pad_to_shape,
                        num_classes, processor_type, annot_type, num_gpu, **kwargs)

