import os
import functools
import tensorflow as tf
from tensorflow.python.ops import parsing_ops
import random

from protos import input_reader_pb2

_DATASET_SHUFFLE_SEED = random.randint(-(2**32-1),2**32-1)#7

_IMAGE_FIELD            = 'image'
_IMAGE_NAME_FIELD       = 'image_name'
_LABEL_FIELD           = 'labels_class'

def valid_imsize(size):
    return size is not None and len(size) == 2 and size[0] != 0 and size[1] != 0


def decode_image_file(filename, channels, size=None, resize=None,
                        method=tf.image.ResizeMethod.BILINEAR, name=None):

    img_str = tf.read_file(filename)
    img = tf.image.decode_png(img_str, channels=channels, name=name)

    valid_size = valid_imsize(size)
    valid_resize = valid_imsize(resize)
    
    if not valid_resize and valid_size:
        img.set_shape(size)
    elif valid_resize:
        img = tf.image.resize_images(img, resize, method)

    return img

def _create_tf_example_decoder(input_reader_config):

    def decode_example(ex_proto):
        keys_to_features = {
            'image/format':
                tf.FixedLenFeature((), tf.string, default_value='jpeg'),
            'image/filename':
                tf.FixedLenFeature((), tf.string),
            'image/segmentation/filename':
                tf.FixedLenFeature((), tf.string),
            'image/segmentation/class/format':
                tf.FixedLenFeature((), tf.string, default_value='png'),
        }

        height = input_reader_config.height
        width = input_reader_config.width
        rheight = input_reader_config.rheight
        rwidth = input_reader_config.rwidth

        decoded = tf.parse_single_example(ex_proto, keys_to_features)

        input_image = decode_image_file(
            decoded["image/filename"],
            3, (height, width), (rheight, rwidth))
        ground_truth_image = decode_image_file(
            decoded["image/segmentation/filename"],
            1, (height, width), (rheight, rwidth),
            tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        items_to_handlers = {
            _IMAGE_FIELD: input_image,
            _IMAGE_NAME_FIELD: decoded["image/filename"],
            _LABEL_FIELD: ground_truth_image,
        }
        return items_to_handlers

    return decode_example


def build(input_reader_config):
    if not isinstance(input_reader_config, input_reader_pb2.InputReader):
        raise ValueError('input_reader_config not of type '
                         'input_reader_pb2.InputReader.')

    reader_config = input_reader_config.tf_record_input_reader
    if reader_config is None:
        raise ValueError('input_reader_config must have '
                             '`tf_record_input_reader`.')

    if not reader_config.input_path or \
            not os.path.isfile(reader_config.input_path[0]):
        raise ValueError('At least one input path must be specified in '
                         '`input_reader_config`.')

    decoder = _create_tf_example_decoder(input_reader_config)

    dataset = tf.data.TFRecordDataset(reader_config.input_path[:],
                            "GZIP",
                            input_reader_config.num_readers)
    
    dataset = dataset.map(decoder, num_parallel_calls=input_reader_config.num_readers)
    epochs = input_reader_config.num_epochs if input_reader_config.num_epochs > 0 else None
    dataset = dataset.repeat(epochs)
    
    if input_reader_config.shuffle:
        dataset = dataset.shuffle(input_reader_config.queue_capacity,
                                    seed=_DATASET_SHUFFLE_SEED)
        print("shuffle seed:", _DATASET_SHUFFLE_SEED)

    return dataset
