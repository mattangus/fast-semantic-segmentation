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
from post_process.entropy import EntropyProcessor
from post_process.alent import AlEntProcessor
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
    "Entropy": EntropyProcessor,
    "AlEnt": AlEntProcessor
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

def filter_ood(img, thresh=180, dilate=5, erode=5):
    if dilate % 2 == 0:
        dilate -= 1
    if erode % 2 == 0:
        erode -= 1
    if dilate < 1:
        dilate = 1
    if erode < 1:
        erode = 1
    #import pdb; pdb.set_trace()
    edges = cv2.Canny((img*255).astype(np.uint8), thresh, thresh)
    di = cv2.dilate(edges, np.ones((dilate, dilate)))
    er = cv2.erode(di, np.ones((erode, erode)))

    dtform = cv2.distanceTransform(255 - er,
                       distanceType=cv2.DIST_L2,
                       maskSize=cv2.DIST_MASK_PRECISE)

    dtform[dtform > 10] = 10
    dtform /= 10

    border_probs = dtform
    probs = cv2.dilate(border_probs*img, np.ones((3,3)))
    return probs

def run_inference_graph(model, trained_checkpoint_prefix,
                        dataset, num_images, ignore_label, pad_to_shape,
                        num_classes, processor_type, annot_type, num_gpu,
                        export_folder, **kwargs):
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
    #weights = processor.get_weights()
    #ood_mean = tf.reduce_sum(ood_score*weights)/tf.reduce_sum(weights)
    #ood_median = get_median(ood_score)
    #pct_ood_gt = tf.reduce_sum(processor.annot*weights)/tf.reduce_sum(weights)
    #point_list = []
    roc_points = processor.metrics["roc"]
    iou_points = processor.metrics["iou"]
    threshs = np.array(range(400))/(400-1)
    #######################################

    moose_mask = cv2.imread("imgs/moose_mask.png")[...,0:1]

    moose_mask = (moose_mask > 128).astype(np.uint8)

    feed = processor.get_feed_dict()
    prediction = processor.get_prediction()
    colour_prediction = _map_to_colored_labels(prediction, OOD_LABEL_COLORS)
    colour_annot = _map_to_colored_labels(annot_pl, OOD_LABEL_COLORS)

    num_step = num_images // batch

    # previous_export_set = set([os.path.basename(f) for f in glob("exported/*/*/*.png")])
    #previous_export_set = {'sun_bccbrnzxuvtlnfte.png', 'sun_btotndklvjecpext.png', '05_Schafgasse_1_000015_000150_leftImg8bit.png', '07_Festplatz_Flugfeld_000000_000250_leftImg8bit.png', 'sun_bsxsdrjnkydomeni.png', 'frankfurt_000001_071288_leftImg8bit.png', '02_Hanns_Klemm_Str_44_000001_000200_leftImg8bit.png', '04_Maurener_Weg_8_000002_000140_leftImg8bit.png', 'rand.png', '05_Schafgasse_1_000004_000170_leftImg8bit.png', 'munster_000040_000019_leftImg8bit.png', 'sun_bbcoqwpogowtuyvw.png', '02_Hanns_Klemm_Str_44_000005_000190_leftImg8bit.png', '07_Festplatz_Flugfeld_000001_000230_leftImg8bit.png', '07_Festplatz_Flugfeld_000002_000440_leftImg8bit.png', '04_Maurener_Weg_8_000005_000200_leftImg8bit.png', 'munster_000074_000019_leftImg8bit.png', '04_Maurener_Weg_8_000008_000200_leftImg8bit.png', 'frankfurt_000001_049770_leftImg8bit.png', 'sun_aaalbzqrimafwbiv.png', '02_Hanns_Klemm_Str_44_000015_000210_leftImg8bit.png', 'sun_aevmsxcxjbsoluch.png', 'sun_bgboysxblgxwcinn.png', 'sun_bjvurbfklntazktu.png', '04_Maurener_Weg_8_000012_000190_leftImg8bit.png', '02_Hanns_Klemm_Str_44_000011_000240_leftImg8bit.png', '02_Hanns_Klemm_Str_44_000009_000220_leftImg8bit.png', '04_Maurener_Weg_8_000013_000230_leftImg8bit.png', 'sun_bcebhcwjetrpvgsz.png', 'sun_bgwmloggfpvwqzzr.png', '04_Maurener_Weg_8_000000_000200_leftImg8bit.png', 'sun_blpteetxpjmjcejm.png', '07_Festplatz_Flugfeld_000003_000340_leftImg8bit.png', '12_Umberto_Nobile_Str_000001_000280_leftImg8bit.png', '07_Festplatz_Flugfeld_000003_000320_leftImg8bit.png', '05_Schafgasse_1_000012_000220_leftImg8bit.png', 'sun_bcqjcrtydolfnxqd.png', 'sun_bvhyciwhwphjbpjz.png', '04_Maurener_Weg_8_000003_000130_leftImg8bit.png', '02_Hanns_Klemm_Str_44_000014_000200_leftImg8bit.png', '04_Maurener_Weg_8_000004_000210_leftImg8bit.png', '04_Maurener_Weg_8_000008_000180_leftImg8bit.png', 'sun_aaaenaoynzhoyheo.png', 'sun_aqvldktdprlskoki.png', 'sun_bjlpzthlefdpouad.png', 'lindau_000016_000019_leftImg8bit.png', 'frankfurt_000001_025921_leftImg8bit.png', '07_Festplatz_Flugfeld_000000_000260_leftImg8bit.png'}
    previous_export_set = set()
    print(previous_export_set)

    all_results = []
    category_results = {}

    config = tf.ConfigProto(allow_soft_placement=True)
    # config.gpu_options.per_process_gpu_memory_fraction=0.8
    run_options = tf.RunOptions(report_tensor_allocations_upon_oom = True)
    config.gpu_options.allow_growth = True
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

            cur_path = image_path[0].decode()
            save_name = cur_path.replace("/mnt/md0/Data/", "").replace(".jpg", ".png")
            if ".png" not in save_name:
                save_name += ".png"
            previous_export = save_name in previous_export_set
            # if not previous_export:
            #     print("skipping")
            #     continue

            # from libs import perlin
            # h,w = placeholder_tensor.shape.as_list()[1:3]
            # #img_raw = np.expand_dims((perlin.make_perlin(10,h,w)*255).astype(np.uint8),0)
            # m = np.mean(img_raw,(0,1,2))
            # s = np.std(img_raw,(0,1,2))
            # _channel_means = [123.68, 116.779, 103.939]
            # norm = np.clip(img_raw - m + _channel_means,0,255)
            #img_raw = norm
            #import pdb; pdb.set_trace()

            # img_raw *= moose_mask

            if preprocess_input is not None:
                processed_input = sess.run(preprocess_input, feed_dict={placeholder_tensor: img_raw, annot_pl: annot_raw, name_pl: image_path})
            else:
                processed_input = img_raw

            # ood1 = sess.run(ood_score, feed_dict={placeholder_tensor: processed_input, annot_pl: annot_raw, name_pl: image_path})
            # ood2 = sess.run(ood_score, feed_dict={placeholder_tensor: img_raw, annot_pl: annot_raw, name_pl: image_path})
            # plt.figure()
            # plt.imshow(ood1[0,...,0])
            # plt.figure()
            # plt.imshow(ood2[0,...,0])
            # plt.show()

            feed_dict = {
                placeholder_tensor: processed_input,
                annot_pl: annot_raw,
                name_pl: image_path
            }

            feed_dict.update(feed)

            # all_pred_do = []
            # for tempi in range(2):
            #     all_pred_do.append(sess.run(processor.stacked_pred, feed_dict))

            sess.run(reset_metric)

            res = {}
            for f in fetch:
                #print("running", f)
                res.update(sess.run(f, feed_dict, options=run_options))
            
            roc, iou = sess.run([roc_points, iou_points])

            result = processor.post_process(res)

            all_results.append([save_name, result["auroc"],result["aupr"],result["max_iou"],result["fpr_at_tpr"],result["detection_error"]])
            
            # category = save_name.replace("SUN2012/Images/","").split("/")[1]
            # print(idx/num_step,":", category, "          ", end="\r")

            # intresting_result = np.sum(np.logical_and(annot_raw >= 19, annot_raw != 255))/np.prod(annot_raw.shape) > 0.005

            # if intresting_result:
            #     if category not in category_results:
            #         category_results[category] = {"auroc": [], "aupr": [], "max_iou": [], "fpr_at_tpr": [], "detection_error": []}
            #     category_results[category]["auroc"].append(result["auroc"])
            #     category_results[category]["aupr"].append(result["aupr"])
            #     category_results[category]["max_iou"].append(result["max_iou"])
            #     category_results[category]["fpr_at_tpr"].append(result["fpr_at_tpr"])
            #     category_results[category]["detection_error"].append(result["detection_error"])

            # cur_point = sess.run([pct_ood_gt, ood_mean, ood_median], feed_dict)
            # print(cur_point)

            # point_list.append(cur_point)
            # print(result["auroc"], np.sum(np.logical_and(annot_raw >= 19, annot_raw != 255))/np.prod(annot_raw.shape))

            # intresting_result = result["auroc"] > 0.9 or (result["auroc"] > 0.0001 and result["auroc"] < 0.1)
            # intresting_result = np.sum(np.logical_and(annot_raw >= 19, annot_raw != 255))/np.prod(annot_raw.shape) > 0.005

            previous_export = False
            if True or previous_export:
                output_image, new_annot, colour_pred = sess.run([ood_score, colour_annot, colour_prediction], feed_dict, options=run_options)

                if len(output_image.shape) == 3:
                    output_image = np.expand_dims(output_image,-1)

                # output_image -= output_image.min()
                # output_image /= output_image.max()

                out_img = img_raw[0][..., ::-1].astype(np.uint8)
                out_pred = colour_pred[0][..., ::-1].astype(np.uint8)
                out_map = output_image[0,...,0]
                # plt.imshow(out_map)
                # plt.show()
                # import pdb; pdb.set_trace()
                #{"mean_sub": processor.mean_sub,"img_dist": processor.img_dist,"bad_pixel": processor.bad_pixel,"var_inv_tile": processor.var_inv_tile,"left": processor.left}
                out_annot = new_annot[0][..., ::-1].astype(np.uint8)

                # iou_i = np.argmax(iou)
                # fpr, tpr = roc[:,0], roc[:,1]
                # roc_i = np.argmax(tpr + 1 - fpr)
                # iou_t = threshs[iou_i]
                # roc_t = threshs[roc_i]

                # # roc_select = ((output_image[0,...,0]) > roc_t).astype(np.uint8)*255
                # # iou_select = ((output_image[0,...,0]) > iou_t).astype(np.uint8)*255

                # overlay = cv2.addWeighted(out_pred, 0.5, out_img, 0.5, 0)

                # cv2.imshow("image", cv2.resize(out_img, (0,0), fx=0.9, fy=0.9))
                # cv2.imshow("uncertainty", cv2.resize(out_map, (0,0), fx=0.9, fy=0.9))
                # cv2.imshow("annot", cv2.resize(out_annot, (0,0), fx=0.9, fy=0.9))
                # cv2.imshow("prediction", cv2.resize(overlay, (0,0), fx=0.9, fy=0.9))

                print(save_name)

                def do_save():
                    save_folder = os.path.join(export_folder, processor.name)
                    img_save_path = os.path.join(save_folder, "image")
                    map_save_path = os.path.join(save_folder, "map")
                    pred_save_path = os.path.join(save_folder, "pred")
                    annot_save_path = os.path.join(save_folder, "annot")
                    
                    # roc_save_path = os.path.join(save_folder, "roc")
                    # iou_save_path = os.path.join(save_folder, "iou")
                    # for f in [img_save_path, map_save_path, pred_save_path, annot_save_path, roc_save_path, iou_save_path]:
                    for f in [img_save_path, map_save_path, pred_save_path, annot_save_path]:
                        os.makedirs(os.path.join(f, os.path.dirname(save_name)), exist_ok=True)
                    s1 = cv2.imwrite(os.path.join(img_save_path, save_name), out_img)
                    s2 = cv2.imwrite(os.path.join(map_save_path, save_name.replace(".png", ".exr")), out_map)
                    s3 = cv2.imwrite(os.path.join(pred_save_path, save_name), out_pred)
                    s4 = cv2.imwrite(os.path.join(annot_save_path, save_name), out_annot)
                    if not (s1 and s2 and s3 and s4):
                        import pdb; pdb.set_trace()
                    # cv2.imwrite(os.path.join(roc_save_path, save_name), roc_select)
                    # cv2.imwrite(os.path.join(iou_save_path, save_name), iou_select)

                do_save()
                # if previous_export:
                #     do_save()
                #     #previous_export_set.remove(save_name)
                #     if len(previous_export_set) == 0:
                #         break
                # else: #let us decide
                #     while True:
                #         key = cv2.waitKey()
                #         if key == 27: #escape
                #             return
                #         elif key == 32: #space
                #             break
                #         elif key == 115: #s
                #             do_save()
                #             print("saved!")
                #         elif key == 98: #b
                #             import pdb; pdb.set_trace()

        # print()
        # csv_file_name = "category_score/" + processor.name + ".csv"
        # os.makedirs("category_score", exist_ok=True)
        # with open(csv_file_name, "w") as csv:
        #     csv.write("category,auroc,aupr,max_iou,fpr_at_tpr,detection_error,count\n")
        #     for c in sorted(list(category_results.keys())):
        #         csv.write(c + ",")
        #         for metric_name in ["auroc","aupr","max_iou","fpr_at_tpr","detection_error"]:
        #             csv.write(str(np.mean(category_results[c][metric_name])) + ",")
        #         csv.write(str(len(category_results[c]["auroc"])) + "\n")

        meta = os.path.join(export_folder, processor.name, "meta.csv")
        with open(meta, "w") as f:
            f.write("path,auroc,aupr,max_iou,fpr_at_tpr,detection_error\n")
            f.write("\n".join([",".join(map(str, l)) for l in all_results]))

        # points = np.array(point_list)
        # plt.scatter(points[:,0], points[:,1])
        # plt.show()
        # import pdb; pdb.set_trace()
        # print("here")

def extract_images(gpus, model_config, data_config,
                    trained_checkpoint, pad_to_shape,
                    processor_type, annot_type, is_debug,
                    export_folder, **kwargs):

    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    pipeline_config = read_config(model_config, data_config)

    if pad_to_shape is not None and isinstance(pad_to_shape, str) :
        pad_to_shape = [
            int(dim) if dim != '-1' else None
                for dim in pad_to_shape.split(',')]

    input_reader = pipeline_config.input_reader
    input_reader.shuffle = False
    if len(input_reader.tf_record_input_reader) > 1:
        input_reader.tf_record_input_reader.pop()
        print("REMOVED INPUT READER:\n", input_reader)
    ignore_label = input_reader.ignore_label

    num_classes, segmentation_model = model_builder.build(
        pipeline_config.model, is_training=False, ignore_label=ignore_label)
    with tf.device("cpu:0"):
        dataset = dataset_builder.build(input_reader, 1)

    num_gpu = len(gpus.split(","))

    num_examples = sum([r.num_examples for r in input_reader.tf_record_input_reader])

    run_inference_graph(segmentation_model, trained_checkpoint, dataset,
                        num_examples, ignore_label, pad_to_shape,
                        num_classes, processor_type, annot_type, num_gpu,
                        export_folder, **kwargs)

