r"""Run inference on an image or group of images."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import timeit
import numpy as np
import tensorflow as tf
from google.protobuf import text_format
from numpy.core.umath_tests import inner1d
import matplotlib.pyplot as plt
import cv2
from sklearn.decomposition import PCA
#from sklearn.manifold import TSNE
from MulticoreTSNE import MulticoreTSNE as TSNE
from mpl_toolkits.mplot3d import Axes3D
import random
import colorsys
from multiprocessing import Process, Queue, Pool
import json

from protos import pipeline_pb2
from builders import model_builder, dataset_builder
from libs.exporter import deploy_segmentation_inference_graph, _map_to_colored_labels
from libs.constants import CITYSCAPES_LABEL_COLORS, CITYSCAPES_LABEL_IDS, URSA_LABEL_COLORS
from libs.metrics import mean_iou
from libs.ursa_map import train_name
from libs import stat_computer as stats

import scipy.stats as st

#from third_party.mem_gradients_patch import gradients

slim = tf.contrib.slim

flags = tf.app.flags

FLAGS = flags.FLAGS

prefetch_queue = slim.prefetch_queue

flags.DEFINE_string('input_shape', '1024,2048,3', # default Cityscapes values
                    'The shape to use for inference. This should '
                    'be in the form [height, width, channels]. A batch '
                    'dimension is not supported for this test script.')

flags.DEFINE_string('patch_size', None, '')

flags.DEFINE_string('pad_to_shape', '1025,2049', # default Cityscapes values
                     'Pad the input image to the specified shape. Must have '
                     'the shape specified as [height, width].')

flags.DEFINE_string('config_path', None,
                    'Path to a pipeline_pb2.TrainEvalPipelineConfig config '
                    'file.')

flags.DEFINE_string('trained_checkpoint', None,
                    'Path to trained checkpoint, typically of the form '
                    'path/to/model.ckpt')

flags.DEFINE_string('eval_dir', None, 'Path to write outputs images.')

flags.DEFINE_string('split', None, 'either "train" or "eval" or "ood"')

flags.DEFINE_boolean('label_ids', False,
                     'Whether the output should be label ids.')

flags.DEFINE_boolean('debug', False,'')

def create_input(tensor_dict,
                batch_size,
                batch_queue_capacity,
                batch_queue_threads,
                prefetch_queue_capacity):

    def cast_and_reshape(tensor_dict, dicy_key):
        items = tensor_dict[dicy_key]
        float_images = tf.to_float(items)
        tensor_dict[dicy_key] = float_images
        return tensor_dict

    tensor_dict = cast_and_reshape(tensor_dict,
                    dataset_builder._IMAGE_FIELD)

    batched_tensors = tf.train.batch(tensor_dict,
        batch_size=batch_size, num_threads=batch_queue_threads,
        capacity=batch_queue_capacity, dynamic_pad=True,
        allow_smaller_final_batch=False)

    return prefetch_queue.prefetch_queue(batched_tensors,
        capacity=prefetch_queue_capacity,
        dynamic_pad=False)

def to_img(x):
    if len(x.shape) == 2:
        x = np.expand_dims(x, -1)
    return (x/np.max(x)*255).astype(np.uint8)

def create_tf_example(input_tensor, annot_tensor, final_logits, unscaled_logits, input_name):
    
    def _bytes_feature(values):
        return tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[values]))

    def _int64_feature(values):
        if not isinstance(values, (tuple, list)):
            values = [values]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

    def tensor_to_example(tensor, name):
        ret_dict = {
            name + "/data": _bytes_feature(tensor.tostring())
        }
        return ret_dict

    feature_dict = {
        "filename": _bytes_feature(input_name)
    }
    
    # input_res = global_pool.apipe(tensor_to_example, input_tensor, "input")
    # annot_res = global_pool.apipe(tensor_to_example, annot_tensor, "annot")
    # pred_res = global_pool.apipe(tensor_to_example, pred_tensor, "pred")
    # final_res = global_pool.apipe(tensor_to_example, final_logits, "final_logits")
    # unscaled_res = global_pool.apipe(tensor_to_example, unscaled_logits, "unscaled_logits")

    # feature_dict.update(input_res.get())
    # feature_dict.update(annot_res.get())
    # feature_dict.update(pred_res.get())
    # feature_dict.update(final_res.get())
    # feature_dict.update(unscaled_res.get())

    feature_dict.update(tensor_to_example(input_tensor, "input"))
    feature_dict.update(tensor_to_example(annot_tensor, "annot"))
    feature_dict.update(tensor_to_example(final_logits, "final_logits"))
    feature_dict.update(tensor_to_example(unscaled_logits, "unscaled_logits"))
    # feature_dict.update(tensor_to_example(grads, "grads"))

    example = tf.train.Example(
        features=tf.train.Features(feature=feature_dict))
    return example.SerializeToString()

def get_tensor_config(tensor):
    return {
        "dtype": tensor.dtype.name,
        "shape": tensor.shape.as_list()
    }

def tf_serialize_example(f0,f1,f2,f3,f4):
    tf_string = tf.py_func(
        create_tf_example, 
        (f0,f1,f2,f3,f4),  # pass these args to the above function.
        tf.string)      # the return type is <a href="../../api_docs/python/tf#string"><code>tf.string</code></a>.
    return tf.reshape(tf_string, ()) # The result is a scalar

def write_thread(writer, queue):
    while True:
        record = queue.get()
        if record is None:
            break
        writer.write(record)
    print("thread complete")

def write_example(file, data):
    with open(file, "wb") as f:
        f.write(data)

class ParallelWriter(object):
    
    def __init__(self, filename, queue_size=32):
        self.filename = filename
        options = tf.io.TFRecordOptions(compression_type=tf.io.TFRecordCompressionType.GZIP)
        self.record_writer = tf.io.TFRecordWriter(filename, options)

        self.write_queue = Queue(queue_size)
        self.p = Process(target=write_thread, args=(self.record_writer, self.write_queue))
        self.p.start()
    
    def put(self, data):
        self.write_queue.put(data)
    
    def close(self):
        self.put(None)
        self.p.join()
        self.record_writer.flush()
        self.record_writer.close()
        print("closed writer")
    
    def size(self):
        return self.write_queue.qsize()

def dataset_to_config(dataset):
    shapes = dataset.output_shapes
    types = dataset.output_types

    config = {
        "input": {
            "dtype": types[0].name,
            "shape": shapes[0].as_list(),
        },
        "annot": {
            "dtype": types[1].name,
            "shape": shapes[1].as_list(),
        },
        "final_logits": {
            "dtype": types[2].name,
            "shape": shapes[2].as_list(),
        },
        "unscaled_logits": {
            "dtype": types[3].name,
            "shape": shapes[3].as_list(),
        },
        # "grads": {
        #     "dtype": types[4].name,
        #     "shape": shapes[4].as_list(),
        # }
    }
    return config
def run_inference_graph(model, trained_checkpoint_prefix,
                        input_dict, num_images, input_shape, pad_to_shape,
                        label_color_map, num_classes, eval_dir):
    assert len(input_shape) == 3, "input shape must be rank 3"
    batch = 4
    effective_shape = [batch] + input_shape

    input_queue = create_input(input_dict, batch, 15, 15, 15)
    input_dict = input_queue.dequeue()

    input_tensor = input_dict[dataset_builder._IMAGE_FIELD]
    annot_tensor = input_dict[dataset_builder._LABEL_FIELD]
    input_name = input_dict[dataset_builder._IMAGE_NAME_FIELD]

    outputs, _ = deploy_segmentation_inference_graph(
        model=model,
        input_shape=effective_shape,
        input=input_tensor,
        pad_to_shape=pad_to_shape,
        input_type=tf.float32)

    #pred_tensor = outputs[model.main_class_predictions_key]
    final_logits = outputs[model.final_logits_key]
    unscaled_logits = outputs[model.unscaled_logits_key]

    # grads = tf.gradients(unscaled_logits, input_tensor)
    # eps = tf.placeholder(tf.float32, (), "eps")
    # adv_img = input_tensor - eps*tf.sign(grads)

    # import pdb; pdb.set_trace()

    feats_dir = os.path.join(eval_dir, "feats")
    os.makedirs(feats_dir, exist_ok=True)
    out_record = os.path.join(feats_dir, "feats_" + FLAGS.split + ".record")

    config_file = out_record + ".config"
    features_dataset = tf.data.Dataset.from_tensor_slices((input_tensor, annot_tensor, final_logits, unscaled_logits, input_name))
    config = dataset_to_config(features_dataset)
    features_dataset = features_dataset.map(tf_serialize_example, batch*2)

    with open(config_file, "w") as f:
        json.dump(config, f)

    serialize_iter = features_dataset.make_initializable_iterator()
    init = serialize_iter.initializer
    next_elem = serialize_iter.get_next()

    # pool = Pool(32)

    queue_size = 36
    num_shards = 6
    writers = [ParallelWriter(out_record + "-" + str(i), queue_size//num_shards) for i in range(num_shards)]

    num_step = (num_images // batch) * batch

    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        tf.train.start_queue_runners(sess)
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(sess, trained_checkpoint_prefix)

        for idx in range(num_step):
            start_time = timeit.default_timer()
            if idx % batch == 0:
                sess.run(init)
            
            data = sess.run(next_elem)
            writers[idx % num_shards].put(data)

            elapsed = timeit.default_timer() - start_time
            # print('{0:.4f} iter: {1}, size: {2}'.format(elapsed, idx+1, write_queue.qsize()))
            print('{0:.4f} iter: {1}, size: {2}'.format(elapsed, idx+1, sum([w.size() for w in writers])))

        # write_queue.put(None)
        # p.join()
        for w in writers:
            w.close()
        
def main(_):
    eval_dir = FLAGS.eval_dir

    pipeline_config = pipeline_pb2.PipelineConfig()
    with tf.gfile.GFile(FLAGS.config_path, 'r') as f:
        text_format.Merge(f.read(), pipeline_config)

    pad_to_shape = None
    if FLAGS.input_shape:
        input_shape = [
            int(dim) if dim != '-1' else None
                for dim in FLAGS.input_shape.split(',')]
    else:
        raise ValueError('Must supply `input_shape`')

    if FLAGS.pad_to_shape:
        pad_to_shape = [
            int(dim) if dim != '-1' else None
                for dim in FLAGS.pad_to_shape.split(',')]

    label_map = (CITYSCAPES_LABEL_IDS
        if FLAGS.label_ids else CITYSCAPES_LABEL_COLORS)

    num_classes, segmentation_model = model_builder.build(
        pipeline_config.model, is_training=False)

    if FLAGS.split == "ood":
        input_reader = pipeline_config.ood_eval_input_reader
    elif FLAGS.split == "eval":
        input_reader = pipeline_config.eval_input_reader
    elif FLAGS.split == "train":
        input_reader = pipeline_config.train_input_reader
    else:
        raise ValueError("dataset")
    
    input_reader.num_epochs = 1
    input_dict = dataset_builder.build(input_reader)

    run_inference_graph(segmentation_model, FLAGS.trained_checkpoint,
                        input_dict, input_reader.num_examples, input_shape, pad_to_shape,
                        label_map, num_classes, eval_dir)

if __name__ == '__main__':
    tf.app.run()
