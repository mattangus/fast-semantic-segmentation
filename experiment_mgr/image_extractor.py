r"""Run inference on an image or group of images."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import timeit
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import sys
import logging
import traceback
from io import StringIO
from glob import glob

from builders import model_builder, dataset_builder
from post_process.mahalanobis import MahalProcessor
from post_process.max_softmax import MaxSoftmaxProcessor
from post_process.droput import DropoutProcessor
from post_process.confidence import ConfidenceProcessor
from protos.config_reader import read_config
from libs.exporter import deploy_segmentation_inference_graph, _map_to_colored_labels
from libs.constants import OOD_LABEL_COLORS

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
    "ODIN": MaxSoftmaxProcessor,
    "Dropout": DropoutProcessor,
    "Confidence": ConfidenceProcessor,
}


def get_median(v):
    v = tf.reshape(v, [-1])
    l = v.get_shape()[0]
    mid = l//2 + 1
    val = tf.nn.top_k(v, mid).values
    if l % 2 == 1:
        return val[-1]
    else:
        return 0.5 * (val[-1] + val[-2])

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

    processor.name = processor_type
    processor.post_process_ops()

    preprocess_input = processor.get_preprocessed()

    input_fetch = [input_name, input_tensor, annot_tensor]

    metric_vars = [v for v in tf.local_variables() if "ConfMat" in v.name]
    reset_metric = tf.variables_initializer(metric_vars)

    fetch = processor.get_fetch_dict()
    ood_score = processor.get_output_image()

    #######################################
    weights = processor.get_weights()
    ood_mean = tf.reduce_sum(ood_score*weights)/tf.reduce_sum(weights)
    ood_median = get_median(ood_score)
    pct_ood_gt = tf.reduce_sum(processor.annot*weights)/tf.reduce_sum(weights)
    point_list = []
    #######################################

    feed = processor.get_feed_dict()
    prediction = processor.get_prediction()
    colour_prediction = _map_to_colored_labels(prediction, OOD_LABEL_COLORS)
    colour_annot = _map_to_colored_labels(annot_pl, OOD_LABEL_COLORS)

    num_step = num_images // batch

    previous_export_set = set([os.path.basename(f) for f in glob("exported/*/*/*.png")])
    print(previous_export_set)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction=0.8
    run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
    #config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        init_feed = processor.get_init_feed()
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()],init_feed)

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

            m = np.mean(img_raw,(0,1,2))
            s = np.std(img_raw,(0,1,2))
            _channel_means = [123.68, 116.779, 103.939]
            norm = np.clip(img_raw - m + _channel_means,0,255)
            img_raw = norm

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

            sess.run(reset_metric)

            res = {}
            for f in fetch:
                #print("running", f)
                res.update(sess.run(f, feed_dict, options=run_options))

            result = processor.post_process(res)

            cur_point = sess.run([pct_ood_gt, ood_mean, ood_median], feed_dict)
            print(cur_point)

            point_list.append(cur_point)
            # print(result["auroc"], np.sum(np.logical_and(annot_raw >= 19, annot_raw != 255))/np.prod(annot_raw.shape))

            # intresting_result = result["auroc"] > 0.9 or (result["auroc"] > 0.0001 and result["auroc"] < 0.1)
            # intresting_result &= np.sum(np.logical_and(annot_raw >= 19, annot_raw != 255))/np.prod(annot_raw.shape) > 0.005
            # cur_path = image_path[0].decode()
            # save_name = os.path.basename(cur_path).replace(".jpg", ".png")
            # if ".png" not in save_name:
            #     save_name += ".png"
            # previous_export = save_name in previous_export_set


            # previous_export = False
            # if True or previous_export:
            #     output_image, new_annot, colour_pred = sess.run([ood_score, colour_annot, colour_prediction], feed_dict, options=run_options)

            #     if len(output_image.shape) == 3:
            #         output_image = np.expand_dims(output_image,-1)

            #     # output_image -= output_image.min()
            #     # output_image /= output_image.max()

            #     out_img = img_raw[0][..., ::-1]
            #     out_pred = colour_pred[0][..., ::-1].astype(np.uint8)
            #     out_map = cv2.applyColorMap((output_image[0,...,0]*255).astype(np.uint8), cv2.COLORMAP_JET)
            #     out_annot = new_annot[0][..., ::-1].astype(np.uint8)

            #     cv2.imshow("image", cv2.resize(out_img, (0,0), fx=0.8, fy=0.8))
            #     cv2.imshow("uncertainty", cv2.resize(out_map, (0,0), fx=0.8, fy=0.8))
            #     cv2.imshow("annot", cv2.resize(out_annot, (0,0), fx=0.8, fy=0.8))
            #     cv2.imshow("prediction", cv2.resize(out_pred, (0,0), fx=0.8, fy=0.8))

            #     print(cur_path)

            #     def do_save():
            #         save_folder = "exported/" + processor.name
            #         img_save_path = os.path.join(save_folder, "image")
            #         map_save_path = os.path.join(save_folder, "map")
            #         pred_save_path = os.path.join(save_folder, "pred")
            #         annot_save_path = os.path.join(save_folder, "annot")
            #         for f in [img_save_path, map_save_path, pred_save_path, annot_save_path]:
            #             os.makedirs(f, exist_ok=True)
            #         cv2.imwrite(os.path.join(img_save_path, save_name), out_img)
            #         cv2.imwrite(os.path.join(map_save_path, save_name), out_map)
            #         cv2.imwrite(os.path.join(pred_save_path, save_name), out_pred)
            #         cv2.imwrite(os.path.join(annot_save_path, save_name), out_annot)

            #     if previous_export:
            #         print("previous export")
            #         do_save()
            #         previous_export_set.remove(save_name)
            #         if len(previous_export_set) == 0:
            #             break
            #     else: #let us decide
            #         while True:
            #             key = cv2.waitKey()
            #             if key == 27: #escape
            #                 return
            #             elif key == 32: #space
            #                 break
            #             elif key == 115: #s
            #                 do_save()
            #                 print("saved!")
            #             elif key == 98: #b
            #                 import pdb; pdb.set_trace()
        
        points = np.array(point_list)
        plt.scatter(points[:,0], points[:,1])
        plt.show()
        import pdb; pdb.set_trace()
        print("here")

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

