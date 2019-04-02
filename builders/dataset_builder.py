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
        img = tf.cast(tf.image.resize_images(img, resize, method), tf.uint8)

    return img

def _create_tf_example_decoder(reader_config):

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

        height = reader_config.height
        width = reader_config.width
        rheight = reader_config.rheight
        rwidth = reader_config.rwidth

        decoded = tf.parse_single_example(ex_proto, keys_to_features)

        input_image = decode_image_file(
            decoded["image/filename"],
            3, (height, width), (rheight, rwidth))
        
        match = tf.strings.regex_full_match(decoded["image/segmentation/filename"], r".*\.png")
        def label_decode():
            return decode_image_file(
                decoded["image/segmentation/filename"],
                1, (height, width), (rheight, rwidth),
                tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        def label_value():
            valid_size = valid_imsize((height, width))
            valid_resize = valid_imsize((rheight, rwidth))

            if not valid_resize and valid_size:
                shape = [height, width, 1]
            elif valid_resize:
                shape = [rheight, rwidth, 1]

            ret = tf.ones(shape, dtype=tf.uint8)*254
            # mult = tf.strings.to_number(decoded["image/segmentation/filename"], out_type=tf.int32)
            # mult = tf.cast(mult, tf.uint8)
            return ret

        ground_truth_image = tf.cond(match,true_fn=label_decode,false_fn=label_value)

        #ground_truth_image = tf.Print(ground_truth_image, [tf.reduce_mean(ground_truth_image)])
        items_to_handlers = {
            _IMAGE_FIELD: input_image,
            _IMAGE_NAME_FIELD: decoded["image/filename"],
            _LABEL_FIELD: ground_truth_image,
        }

        return items_to_handlers

    return decode_example

def _build_random(reader_config):
    height = reader_config.height
    width = reader_config.width
    
    if "normal" in reader_config.input_path:
        rand = lambda: (np.clip(np.random.normal(0.5,size=(height, width, 3)), 0.0, 1.0)*255).astype(np.uint8)
        # rand = tf.distributions.Normal(np.ones((1, height, width, 3),dtype=np.float32)*0.5,1.0).sample()
        #rand = tf.random.normal((1, height, width, 3),mean=0.5,stddev=1.0)
        #rand = tf.clip_by_value(rand, 0.0, 1.0)*255
    elif "uniform" in reader_config.input_path:
        rand = lambda: np.random.uniform(0.0, 255.0, size=(height, width, 3)).astype(np.uint8)
        #rand = tf.distributions.Uniform(np.zeros((1, height, width, 3),dtype=np.float32),1.0).sample()
        #rand = tf.random.uniform((1, height, width, 3),0.0, 255.0)
    
    def gen():
        label = (np.ones((height, width, 1))*19).astype(np.uint8)
        for _ in range(reader_config.num_examples):
            items_to_handlers = {
                _IMAGE_FIELD: rand(),
                _IMAGE_NAME_FIELD: "rand",
                _LABEL_FIELD: label,
            }
            yield items_to_handlers

    types = {
        _IMAGE_FIELD: tf.uint8,
        _IMAGE_NAME_FIELD: tf.string,
        _LABEL_FIELD: tf.uint8,
    }

    shapes = {
        _IMAGE_FIELD: tf.TensorShape([height, width, 3]),
        _IMAGE_NAME_FIELD: tf.TensorShape([]),
        _LABEL_FIELD: tf.TensorShape([height, width, 1]),
    }

    dataset = tf.data.Dataset.from_generator(gen, types, shapes)
    dataset = dataset.repeat(reader_config.num_examples)

    return dataset

def _make_dataset(reader_config, num_readers):
    if ".dist" not in reader_config.input_path:

        if not reader_config.input_path or \
                not os.path.isfile(reader_config.input_path):
            raise ValueError('At least one input path must be specified in '
                            '`input_reader_config`.')


        decoder = _create_tf_example_decoder(reader_config)

        dataset = tf.data.TFRecordDataset(reader_config.input_path,
                                "GZIP",
                                num_readers)
        
        dataset = dataset.map(decoder, num_parallel_calls=num_readers)
    else:
        dataset = _build_random(reader_config)
    
    return dataset

def build(input_reader_config, num_epoch):
    if not isinstance(input_reader_config, input_reader_pb2.InputReader):
        raise ValueError('input_reader_config not of type '
                         'input_reader_pb2.InputReader.')

    reader_config = input_reader_config.tf_record_input_reader
    if reader_config is None:
        raise ValueError('input_reader_config must have '
                             '`tf_record_input_reader`.')

    
    if len(reader_config) == 1:
        dataset = _make_dataset(reader_config[0], input_reader_config.num_readers)
    else:
        datasets = list(map(lambda x: _make_dataset(x, input_reader_config.num_readers), reader_config))
        num_examples = [r.num_examples for r in reader_config]
        dataset_choices = []
        for i, n in enumerate(num_examples):
            dataset_choices.extend([i] * n)

        random.shuffle(dataset_choices)
        dataset_choices = np.array(dataset_choices, dtype=np.int64)
        choice_dataset = tf.data.Dataset.from_tensor_slices(dataset_choices)

        dataset = tf.data.experimental.choose_from_datasets(datasets, choice_dataset)

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
