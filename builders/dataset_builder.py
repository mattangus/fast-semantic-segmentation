import os
import functools
import tensorflow as tf
from tensorflow.python.ops import parsing_ops
import random
import numpy as np

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
        shape = list(size) + [channels]
        img.set_shape(shape)
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

def _build_random(input_reader_config):
    height = input_reader_config.height
    width = input_reader_config.width
    
    reader_config = input_reader_config.tf_record_input_reader
    if "normal" in reader_config.input_path[0]:
        rand = lambda: np.clip(np.random.normal(0.5,size=(height, width, 3)), 0.0, 1.0).astype(np.float32)
        # rand = tf.distributions.Normal(np.ones((1, height, width, 3),dtype=np.float32)*0.5,1.0).sample()
        #rand = tf.random.normal((1, height, width, 3),mean=0.5,stddev=1.0)
        #rand = tf.clip_by_value(rand, 0.0, 1.0)*255
    elif "uniform" in reader_config.input_path[0]:
        rand = lambda: np.random.uniform(0.0, 255.0, size=(height, width, 3)).astype(np.float32)
        #rand = tf.distributions.Uniform(np.zeros((1, height, width, 3),dtype=np.float32),1.0).sample()
        #rand = tf.random.uniform((1, height, width, 3),0.0, 255.0)
    
    def gen():
        label = (np.ones((height, width, 1))*19).astype(np.float32)
        for _ in range(input_reader_config.num_examples):
            items_to_handlers = {
                _IMAGE_FIELD: rand(),
                _IMAGE_NAME_FIELD: "rand",
                _LABEL_FIELD: label,
            }
            yield items_to_handlers

    types = {
        _IMAGE_FIELD: tf.float32,
        _IMAGE_NAME_FIELD: tf.string,
        _LABEL_FIELD: tf.float32,
    }

    shapes = {
        _IMAGE_FIELD: tf.TensorShape([height, width, 3]),
        _IMAGE_NAME_FIELD: tf.TensorShape([]),
        _LABEL_FIELD: tf.TensorShape([height, width, 1]),
    }

    dataset = tf.data.Dataset.from_generator(gen, types, shapes)
    dataset = dataset.repeat(input_reader_config.num_examples)

    return dataset

def build(input_reader_config, num_epoch):
    if not isinstance(input_reader_config, input_reader_pb2.InputReader):
        raise ValueError('input_reader_config not of type '
                         'input_reader_pb2.InputReader.')

    reader_config = input_reader_config.tf_record_input_reader
    if reader_config is None:
        raise ValueError('input_reader_config must have '
                             '`tf_record_input_reader`.')

    if ".dist" not in reader_config.input_path[0]:

        if not reader_config.input_path or \
                not os.path.isfile(reader_config.input_path[0]):
            raise ValueError('At least one input path must be specified in '
                            '`input_reader_config`.')


        decoder = _create_tf_example_decoder(input_reader_config)

        dataset = tf.data.TFRecordDataset(reader_config.input_path[:],
                                "GZIP",
                                input_reader_config.num_readers)
        
        dataset = dataset.map(decoder, num_parallel_calls=input_reader_config.num_readers)
    else:
        dataset = _build_random(input_reader_config)
    epochs = num_epoch if num_epoch > 0 else None
    
    if input_reader_config.shuffle:
        sar_fn = tf.data.experimental.shuffle_and_repeat
        sar = sar_fn(input_reader_config.queue_capacity,
                        epochs, seed=_DATASET_SHUFFLE_SEED)
        dataset = dataset.apply(sar)
        print("shuffle seed:", _DATASET_SHUFFLE_SEED)
    else:
        dataset = dataset.repeat(epochs)

    dataset = dataset.prefetch(input_reader_config.prefetch_queue_capacity)

    return dataset
