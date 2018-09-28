r"""Build a TF Record for Cityscapes Semantic Segmentation dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import glob
import io
import json
import os
import numpy as np
import PIL.Image
import cv2

import tensorflow as tf

flags = tf.app.flags
tf.flags.DEFINE_string('input_pattern', '',
                       'Cityscapes dataset root folder.')
tf.flags.DEFINE_string('annot_pattern', '',
                       'Pattern matching input images for Cityscapes.')
tf.flags.DEFINE_string('vec_pattern', '',
                       'Pattern matching border images for Cityscapes.')
tf.flags.DEFINE_string('cityscapes_dir', '',
                       'Pattern matching ground truth images for Cityscapes.')
tf.flags.DEFINE_string('split_type', '',
                       'Type of split: `train`, `test` or `val`.')
tf.flags.DEFINE_string('output_dir', '', 'Output data directory.')

tf.flags.DEFINE_list('resize', None, 'w,h to resize to')

FLAGS = flags.FLAGS


tf.logging.set_verbosity(tf.logging.INFO)


_DEFAULT_PATTEN = {
    'input': '*_leftImg8bit.png',
    'label': '*_gtFine_labelTrainIds.png',
    'vec': '*_gtFine_labelTrainIds.exr'
}

_DEFAULT_DIR = {
    'image': 'leftImg8bit',
    'label': 'gtFine',
    'vec': 'borders'
}


def _bytes_feature(values):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[values]))


def _int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _open_file(full_path):
    # with tf.gfile.GFile(full_path, 'rb') as fid:
    #     encoded_file = fid.read()
    # encoded_file_io = io.BytesIO(encoded_file)
    # image = PIL.Image.open(encoded_file_io)
    # return image, encoded_file
    return cv2.imread(full_path)

def convert_vec(vec):
    w = vec.shape[1]
    h = vec.shape[0]
    xAbs = vec % w
    yAbs = (vec - xAbs) / w

    x = np.arange(0, w, dtype=np.int32)
    y = np.arange(0, h, dtype=np.int32)
    x, y = np.meshgrid(x,y)
    
    xRel = xAbs - x
    yRel = yAbs - y

    return np.stack([xRel.astype(np.int32), yRel.astype(np.int32)], axis=-1)

def create_tf_example(image_path, label_path, vec_path, image_dir='', is_jpeg=False):
    file_format = 'jpeg' if is_jpeg else 'png'
    full_image_path = os.path.join(image_dir, image_path)
    full_label_path = os.path.join(image_dir, label_path)
    full_vec_path = os.path.join(image_dir, vec_path)
    image = cv2.imread(full_image_path)
    label = cv2.imread(full_label_path)
    vec = cv2.imread(full_vec_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

    height = image.shape[0]
    width = image.shape[1]
    if (height != label.shape[0] or width != label.shape[1] or
        height != vec.shape[0] or width != vec.shape[1]):
        raise ValueError('Input and annotated images must have same dims.'
                        'verify the matching pair for {}'.format(full_image_path))

    if FLAGS.resize is not None:
        size = tuple([int(d) for d in FLAGS.resize])
        image = cv2.resize(image, size)
        label = cv2.resize(label, size, interpolation=cv2.INTER_NEAREST)
        vec = cv2.resize(vec, size, interpolation=cv2.INTER_NEAREST)
        height = image.shape[0]
        width = image.shape[1]

    vec = convert_vec(vec)
    
    _, encoded_image = cv2.imencode("." + file_format, image)
    _, encoded_label = cv2.imencode(".png", label)
    
    feature_dict = {
        'image/encoded': _bytes_feature(encoded_image.tostring()),
        'image/filename': _bytes_feature(
                full_image_path.encode('utf8')),
        'image/format': _bytes_feature(
                file_format.encode('utf8')),
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/channels': _int64_feature(3),
        'image/segmentation/class/encoded': _bytes_feature(encoded_label.tostring()),
        'image/segmentation/class/format':_bytes_feature('png'.encode('utf8')),
        'image/vec': _bytes_feature(vec.tostring()),
        'image/vec/format': _bytes_feature('raw'.encode('utf8'))
    }
    
    example = tf.train.Example(
        features=tf.train.Features(feature=feature_dict))
    return example


def _create_tf_record(images, labels, vecs, output_path):
    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    writer = tf.python_io.TFRecordWriter(output_path, options=options)
    for idx, image in enumerate(images):
        if idx % 100 == 0:
            tf.logging.info('On image %d of %d', idx, len(images))
        tf_example = create_tf_example(
            image, labels[idx], vecs[idx], is_jpeg=False)
        writer.write(tf_example.SerializeToString())
    writer.close()
    tf.logging.info('Finished writing!')


def main(_):
    assert FLAGS.output_dir, '`output_dir` missing.'
    assert FLAGS.split_type, '`split_type` missing.'
    assert (FLAGS.cityscapes_dir) or \
           (FLAGS.input_pattern and FLAGS.annot_pattern), \
           'Must specify either `cityscapes_dir` or ' \
           '`input_pattern` and `annot_pattern`.'

    if not tf.gfile.IsDirectory(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)
    train_output_path = os.path.join(FLAGS.output_dir,
        'cityscapes_{}.record'.format(FLAGS.split_type))

    if FLAGS.cityscapes_dir:
        search_image_files = os.path.join(FLAGS.cityscapes_dir,
            _DEFAULT_DIR['image'], FLAGS.split_type, '*', _DEFAULT_PATTEN['input'])
        search_annot_files = os.path.join(FLAGS.cityscapes_dir,
            _DEFAULT_DIR['label'], FLAGS.split_type, '*', _DEFAULT_PATTEN['label'])
        search_vec_files = os.path.join(FLAGS.cityscapes_dir,
            _DEFAULT_DIR['vec'], FLAGS.split_type, '*', _DEFAULT_PATTEN['vec'])
        image_filenames = glob.glob(search_image_files)
        annot_filenames = glob.glob(search_annot_files)
        vec_filenames = glob.glob(search_vec_files)
    else:
        image_filenames = glob.glob(FLAGS.input_pattern)
        annot_filenames = glob.glob(FLAGS.annot_pattern)
        vec_filenames = glob.glob(FLAGS.annot_pattern)
    
    if len(image_filenames) != len(annot_filenames) or len(image_filenames) != len(vec_filenames):
        print("images: ", len(image_filenames))
        print("annot: ", len(annot_filenames))
        print("vecs: ", len(vec_filenames))
        raise ValueError('Supplied patterns do not have image counts.')

    _create_tf_record(
            sorted(image_filenames),
            sorted(annot_filenames),
            sorted(vec_filenames),
            output_path=train_output_path)


if __name__ == '__main__':
    tf.app.run()
