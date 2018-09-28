import os
import functools
import tensorflow as tf
from tensorflow.python.ops import parsing_ops

from protos import input_reader_pb2

slim = tf.contrib.slim

tfexample_decoder = slim.tfexample_decoder

dataset = slim.dataset

dataset_data_provider = slim.dataset_data_provider

_DATASET_SHUFFLE_SEED = 7

_IMAGE_FIELD            = 'image'
_IMAGE_NAME_FIELD       = 'image_name'
_HEIGHT_FIELD           = 'height'
_WIDTH_FIELD            = 'width'
_LABEL_FIELD           = 'labels_class'
_VEC_FIELD           = 'vec'

_ITEMS_TO_DESCRIPTIONS = {
    'image':        ('A color image of varying height and width.'),
    'labels_class': ('A semantic segmentation label whose size matches image.'
                     'Its values range from 0 (background) to num_classes.'),
}

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

def _create_tf_example_decoder():

    keys_to_features = {
        'image/encoded':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format':
            tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/filename':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/height':
            tf.FixedLenFeature((), tf.int64, default_value=0),
        'image/width':
            tf.FixedLenFeature((), tf.int64, default_value=0),
        'image/segmentation/class/encoded':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/segmentation/class/format':
            tf.FixedLenFeature((), tf.string, default_value='png'),
        'image/vec':
            tf.FixedLenFeature((), tf.string),
        'image/vec/format':
            tf.FixedLenFeature((), tf.string, default_value='raw'),
    }

    input_image = tfexample_decoder.Image(
        image_key='image/encoded',
        format_key='image/format',
        shape=(1024, 2048, 3), # CITYSCAPES SPECIFIC
        channels=3)
    ground_truth_image = tfexample_decoder.Image(
        image_key='image/segmentation/class/encoded',
        format_key='image/segmentation/class/format',
        shape=(1024, 2048, 1), # CITYSCAPES SPECIFIC
        channels=1)
    
    vec_data = Raw(
        image_key='image/vec',
        format_key='image/vec/format',
        shape=(1024, 2048, 2), #CITYSCAPSE SPECIFIC
        dtype=tf.int32)

    items_to_handlers = {
        _IMAGE_FIELD: input_image,
        _IMAGE_NAME_FIELD: tfexample_decoder.Tensor('image/filename'),
        _HEIGHT_FIELD: tfexample_decoder.Tensor('image/height'),
        _WIDTH_FIELD: tfexample_decoder.Tensor('image/width'),
        _LABEL_FIELD: ground_truth_image,
        _VEC_FIELD: vec_data,
    }
    
    return tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)


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

    decoder = _create_tf_example_decoder()

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
            seed=_DATASET_SHUFFLE_SEED)

        (image, image_name, height, width, label, vec) = provider.get([_IMAGE_FIELD,
            _IMAGE_NAME_FIELD, _HEIGHT_FIELD, _WIDTH_FIELD, _LABEL_FIELD, _VEC_FIELD])

    return {
        _IMAGE_FIELD: image,
        _IMAGE_NAME_FIELD: image_name,
        _HEIGHT_FIELD: height,
        _WIDTH_FIELD: width,
        _LABEL_FIELD: label,
        _VEC_FIELD: vec
    }
