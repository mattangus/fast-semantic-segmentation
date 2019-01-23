import os
import functools
import tensorflow as tf
from tensorflow.python.ops import parsing_ops
import random
import glob
import numpy as np

from protos import input_reader_pb2

slim = tf.contrib.slim

tfexample_decoder = slim.tfexample_decoder

dataset = slim.dataset

dataset_data_provider = slim.dataset_data_provider

_DATASET_SHUFFLE_SEED = random.randint(-(2**32-1),2**32-1)#7

_IMAGE_FIELD            = 'image'
_IMAGE_NAME_FIELD       = 'image_name'
_HEIGHT_FIELD           = 'height'
_WIDTH_FIELD            = 'width'
_LABEL_FIELD           = 'labels_class'
_DIST_FIELD             = "min_dist"

_ITEMS_TO_DESCRIPTIONS = {
    'image':        ('A color image of varying height and width.'),
    'labels_class': ('A semantic segmentation label whose size matches image.'
                     'Its values range from 0 (background) to num_classes.'),
}

class NumpyFile(tfexample_decoder.ItemHandler):
    """An ItemHandler that decodes a parsed Tensor as an image."""

    def __init__(self,
                eval_dir,
                filename_key=None,
                item_name=None,
                shape=None,
                resize=None,
                method=tf.image.ResizeMethod.BILINEAR,
                dtype=tf.float32):
        if not filename_key:
            filename_key = 'image/filename'
        if not item_name:
            item_name = "arr_0"

        super(NumpyFile, self).__init__([filename_key])
        self._filename_key = filename_key
        self._item_name = item_name
        self._eval_dir = eval_dir
        self._shape = shape
        self._resize = resize
        self._method = method
        self._dtype = dtype

    def tensors_to_item(self, keys_to_tensors):
        """See base class."""
        filename_tensor = keys_to_tensors[self._filename_key]

        return self._decode(filename_tensor)

    def _decode(self, filename_tensor):

        def decode_numpy(path):
            path = path.decode()
            base_name = os.path.basename(path)
            path = os.path.join(self._eval_dir, base_name + ".npy")
            ret = np.load(path).item()[self._item_name]
            return ret
        
        decoder = tf.py_func(decode_numpy, [filename_tensor], tf.float32)
        decoder.set_shape(self._shape)
        if self._resize[0] != 0 and self._resize[1] != 0:    
            decoder = tf.image.resize_images(decoder, self._resize, method=self._method)
            
        return decoder


class Raw(tfexample_decoder.ItemHandler):
    """An ItemHandler that decodes a parsed Tensor as an image."""

    def __init__(self,
                image_key=None,
                format_key=None,
                shape=None,
                dtype=tf.uint8):
        """Initializes the image.

        Args:
        image_key: the name of the TF-Example feature in which the encoded image
            is stored.
        format_key: the name of the TF-Example feature in which the image format
            is stored.
        shape: the output shape of the image as 1-D `Tensor`
            [height, width, channels]. If provided, the image is reshaped
            accordingly. If left as None, no reshaping is done. A shape should
            be supplied only if all the stored images have the same shape.
        dtype: images will be decoded at this bit depth. Different formats
            support different bit depths.
        """
        if not image_key:
            image_key = 'image/encoded'
        if not format_key:
            format_key = 'image/format'

        super(Raw, self).__init__([image_key, format_key])
        self._image_key = image_key
        self._format_key = format_key
        self._shape = shape
        self._dtype = dtype

    def tensors_to_item(self, keys_to_tensors):
        """See base class."""
        image_buffer = keys_to_tensors[self._image_key]
        image_format = keys_to_tensors[self._format_key]

        return self._decode(image_buffer, image_format)

    def _decode(self, image_buffer, image_format):
        """Decodes the image buffer.

        Args:
        image_buffer: The tensor representing the encoded image tensor.
        image_format: The image format for the image in `image_buffer`. If image
            format is `raw`, all images are expected to be in this format, otherwise
            this op can decode a mix of `jpg` and `png` formats.

        Returns:
        A tensor that represents decoded image of self._shape, or
        (?, ?, self._channels) if self._shape is not specified.
        """

        def decode_raw():
            """Decodes a raw image."""
            return parsing_ops.decode_raw(image_buffer, out_type=self._dtype)

        image = decode_raw()

        if self._shape is not None:
            image = tf.reshape(image, self._shape)

        return image

class ImageFile(tfexample_decoder.ItemHandler):
    """An ItemHandler that decodes a parsed Tensor as an image."""

    def __init__(self,
                filename_key=None,
                shape=None,
                resize=(0,0),
                method=tf.image.ResizeMethod.BILINEAR,
                dtype=tf.uint8):
        """Initializes the image.

        Args:
        image_key: the name of the TF-Example feature in which the encoded image
            is stored.
        format_key: the name of the TF-Example feature in which the image format
            is stored.
        shape: the output shape of the image as 1-D `Tensor`
            [height, width, channels]. If provided, the image is reshaped
            accordingly. If left as None, no reshaping is done. A shape should
            be supplied only if all the stored images have the same shape.
        dtype: images will be decoded at this bit depth. Different formats
            support different bit depths.
        """
        if not filename_key:
            filename_key = 'image/filename'

        super(ImageFile, self).__init__([filename_key])
        self._filename_key = filename_key
        self._shape = shape
        self._resize = resize
        self._method = method
        self._dtype = dtype

    def tensors_to_item(self, keys_to_tensors):
        """See base class."""
        filename = keys_to_tensors[self._filename_key]
        # filename = tf.Print(filename, [filename])
        img = self.filename_to_image(filename, self._shape[-1])

        if self._resize[0] != 0 and self._resize[1] != 0:
            img = tf.image.resize_images(img, self._resize, method=self._method)
        else:
            img.set_shape(self._shape)
        
        return img

    def filename_to_image(self, filename, channels, name=None):
        return tf.image.decode_png(tf.read_file(filename), channels=channels, name=name)

def _create_tf_example_decoder(input_reader_config, eval_dir, max_softmax):

    keys_to_features = {
        # 'image/encoded':
        #     tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format':
            tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/filename':
            tf.FixedLenFeature((), tf.string),
        # 'image/height':
        #     tf.FixedLenFeature((), tf.int64, default_value=0),
        # 'image/width':
        #     tf.FixedLenFeature((), tf.int64, default_value=0),
        # 'image/segmentation/class/encoded':
        #     tf.FixedLenFeature((), tf.string, default_value=''),
        # 'image/segmentation/class/format':
        #     tf.FixedLenFeature((), tf.string, default_value='png'),
        'image/segmentation/filename':
            tf.FixedLenFeature((), tf.string),
        # 'image/segmentation/class/encoded': _bytes_feature(encoded_label.tostring()),
        'image/segmentation/class/format':
            tf.FixedLenFeature((), tf.string, default_value='png'),
    }

    height = input_reader_config.height
    width = input_reader_config.width
    rheight = input_reader_config.rheight
    rwidth = input_reader_config.rwidth

    # input_image = filename_to_image(tfexample_decoder.Tensor('image/filename'), 3)
    # ground_truth_image = filename_to_image(tfexample_decoder.Tensor('image/segmentation/filename'), 3)

    input_image = ImageFile(
        filename_key='image/filename',
        shape=(height, width, 3),
        resize=(rheight, rwidth))
    ground_truth_image = ImageFile(
        filename_key='image/segmentation/filename',
        shape=(height, width, 1),
        resize=(rheight, rwidth),
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    if max_softmax:
        item_name = "unscaled_logits"
    else:
        item_name = "dist"
    test_shape_file = next(glob.iglob(os.path.join(eval_dir, "*.jpg.npy")))
    min_dist = np.load(test_shape_file).item()[item_name]
    shape = min_dist.shape

    min_dist = NumpyFile(eval_dir,
        filename_key="image/filename",
        item_name=item_name,
        shape=shape,
        resize=(rheight, rwidth))

    # input_image = tfexample_decoder.Image(
    #     image_key='image/encoded',
    #     format_key='image/format',
    #     shape=(height, width, 3), # CITYSCAPES SPECIFIC
    #     channels=3)
    # ground_truth_image = tfexample_decoder.Image(
    #     image_key='image/segmentation/class/encoded',
    #     format_key='image/segmentation/class/format',
    #     shape=(height, width, 1), # CITYSCAPES SPECIFIC
    #     channels=1)

    items_to_handlers = {
        _IMAGE_FIELD: input_image,
        _IMAGE_NAME_FIELD: tfexample_decoder.Tensor('image/filename'),
        # _HEIGHT_FIELD: tfexample_decoder.Tensor('image/height'),
        # _WIDTH_FIELD: tfexample_decoder.Tensor('image/width'),
        _LABEL_FIELD: ground_truth_image,
        _DIST_FIELD: min_dist,
    }
    
    return tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)


def build(input_reader_config, eval_dir, max_softmax):
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

    decoder = _create_tf_example_decoder(input_reader_config, eval_dir, max_softmax)

    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)

    with tf.variable_scope("Dataset"):
        train_dataset = dataset.Dataset(
            data_sources=reader_config.input_path[:],
            reader=functools.partial(tf.TFRecordReader, options=options),
            decoder=decoder,
            num_samples=input_reader_config.num_examples,
            items_to_descriptions=_ITEMS_TO_DESCRIPTIONS)
    with tf.variable_scope("DataProvider"):
        provider = dataset_data_provider.DatasetDataProvider(
            train_dataset,
            num_readers=input_reader_config.num_readers,
            num_epochs=(input_reader_config.num_epochs
                if input_reader_config.num_epochs else None),
            shuffle=input_reader_config.shuffle,
            common_queue_capacity=150,
            seed=_DATASET_SHUFFLE_SEED)
        print("shuffle seed:", _DATASET_SHUFFLE_SEED)

        (image, image_name, label, min_dist) = provider.get([_IMAGE_FIELD,
            _IMAGE_NAME_FIELD,
            #_HEIGHT_FIELD, _WIDTH_FIELD,
            _LABEL_FIELD, _DIST_FIELD])
    # label = tf.Print(label, ["label max", tf.reduce_max(label)])
    
    return {
        _IMAGE_FIELD: image,
        _IMAGE_NAME_FIELD: image_name,
        # _HEIGHT_FIELD: height,
        # _WIDTH_FIELD: width,
        _LABEL_FIELD: label,
        _DIST_FIELD: min_dist
    }
