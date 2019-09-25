r"""Run inference on an image or group of images."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import timeit
import numpy as np
from PIL import Image
import tensorflow as tf
from google.protobuf import text_format

from protos.config_reader import read_config
from protos import pipeline_pb2
from builders import model_builder
from libs.exporter import deploy_segmentation_inference_graph
from libs.constants import CITYSCAPES_LABEL_COLORS, CITYSCAPES_LABEL_IDS


slim = tf.contrib.slim

flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_string('input_path', None,
                    'Path to an image or a directory of images.')

flags.DEFINE_string('pad_to_shape', '1025,2049', # default Cityscapes values
                     'Pad the input image to the specified shape. Must have '
                     'the shape specified as [height, width].')

flags.DEFINE_string('data_config', None,
                    'Path to a pipeline_pb2.TrainEvalPipelineConfig config '
                    'file.')

flags.DEFINE_string('model_config', None,
                    'Path to a pipeline_pb2.TrainEvalPipelineConfig config '
                    'file.')

flags.DEFINE_string('trained_checkpoint', None,
                    'Path to trained checkpoint, typically of the form '
                    'path/to/model.ckpt')

flags.DEFINE_string('output_dir', None, 'Path to write outputs images.')

flags.DEFINE_boolean('label_ids', False,
                     'Whether the output should be label ids.')


def _valid_file_ext(input_path):
    ext = os.path.splitext(input_path)[-1].upper()
    return ext in ['.JPG', '.JPEG', '.PNG']


def _get_images_from_path(input_path):
    image_file_paths = []
    if os.path.isdir(input_path):
        for dirpath,_,filenames in os.walk(input_path):
            for f in filenames:
                file_path = os.path.abspath(os.path.join(dirpath, f))
                if _valid_file_ext(file_path):
                    image_file_paths.append(file_path)
        if len(image_file_paths) == 0:
            raise ValueError('No images in directory. '
                             'Files must be JPG or PNG')
    else:
        if not _valid_file_ext(input_path):
            raise ValueError('File must be JPG or PNG.')
        image_file_paths.append(input_path)
    return image_file_paths


def run_inference_graph(model, trained_checkpoint_prefix,
                        input_images, pad_to_shape,
                        label_color_map, output_directory):
    outputs, placeholder_tensor = deploy_segmentation_inference_graph(
        model=model,
        input_shape=[None,None,3],
        pad_to_shape=pad_to_shape,
        label_color_map=label_color_map)
    pred_tensor = outputs[model.main_class_predictions_key]

    with tf.Session() as sess:
        input_graph_def = tf.get_default_graph().as_graph_def()
        saver = tf.train.Saver()
        saver.restore(sess, trained_checkpoint_prefix)

        for idx, image_path in enumerate(input_images):
            image_raw = np.array(Image.open(image_path))

            start_time = timeit.default_timer()
            predictions = sess.run(pred_tensor,
                feed_dict={placeholder_tensor: image_raw})
            elapsed = timeit.default_timer() - start_time

            print('{}) wall time: {}'.format(elapsed, idx+1))
            filename = os.path.basename(image_path)
            save_location = os.path.join(output_directory, filename)

            predictions = predictions.astype(np.uint8)
            if len(label_color_map[0]) == 1:
               predictions = np.squeeze(predictions,-1)
            im = Image.fromarray(predictions[0])
            im.save(save_location, "PNG")


def main(_):
    output_directory = FLAGS.output_dir
    tf.gfile.MakeDirs(output_directory)
    pipeline_config = read_config(FLAGS.model_config, FLAGS.data_config)

    input_images = _get_images_from_path(FLAGS.input_path)
    label_map = (CITYSCAPES_LABEL_IDS
        if FLAGS.label_ids else CITYSCAPES_LABEL_COLORS)

    num_classes, segmentation_model = model_builder.build(
        pipeline_config.model, is_training=False, ignore_label=None)

    run_inference_graph(segmentation_model, FLAGS.trained_checkpoint,
                        input_images, pad_to_shape,
                        label_map, output_directory)

if __name__ == '__main__':
    tf.app.run()