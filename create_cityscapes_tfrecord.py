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
import random

import tensorflow as tf

flags = tf.app.flags
tf.flags.DEFINE_string('input_pattern', '',
                       'Cityscapes dataset root folder.')
tf.flags.DEFINE_string('annot_pattern', '',
                       'Pattern matching input images for Cityscapes.')
tf.flags.DEFINE_string('cityscapes_dir', '',
                       'Pattern matching ground truth images for Cityscapes.')
tf.flags.DEFINE_string('list_file', '',
                       'path to csv with each row as <path to rl>,<path to gt>')                       
tf.flags.DEFINE_string('split_type', '',
                       'Type of split: `train`, `test` or `val`.')
tf.flags.DEFINE_string('output_dir', '', 'Output data directory.')
tf.flags.DEFINE_string('name', '', 'output name.')

tf.flags.DEFINE_integer("label_value", None, "value to pass as label")

tf.flags.DEFINE_list('resize', None, 'w,h to resize to')

tf.flags.DEFINE_bool("shuffle", True, "shuffle list")

FLAGS = flags.FLAGS


tf.logging.set_verbosity(tf.logging.INFO)


_DEFAULT_PATTEN = {
    'input': '*_leftImg8bit.png',
    'label': '*_gtFine_labelTrainIds.png',
}

_DEFAULT_DIR = {
    'image': 'leftImg8bit',
    'label': 'gtFine',
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

def create_tf_example(image_path, label_path, image_dir='', is_jpeg=False):
    file_format = 'jpeg' if is_jpeg else 'png'
    full_image_path = os.path.join(image_dir, image_path)
    full_label_path = os.path.join(image_dir, label_path)
    if FLAGS.label_value is not None:
        full_label_path = str(FLAGS.label_value)
    # image = cv2.imread(full_image_path)
    # label = cv2.imread(full_label_path)

    # height = image.shape[0]
    # width = image.shape[1]
    # if (height != label.shape[0] or width != label.shape[1]):
    #     raise ValueError('Input and annotated images must have same dims.'
    #                     'verify the matching pair for {}'.format(full_image_path))

    # if FLAGS.resize is not None:
    #     size = tuple([int(d) for d in FLAGS.resize])
    #     image = cv2.resize(image, size)
    #     label = cv2.resize(label, size, interpolation=cv2.INTER_NEAREST)
    #     height = image.shape[0]
    #     width = image.shape[1]
    
    # _, encoded_image = cv2.imencode("." + file_format, image)
    # _, encoded_label = cv2.imencode(".png", label)
    
    feature_dict = {
        #'image/encoded': _bytes_feature(encoded_image.tostring()),
        'image/filename': _bytes_feature(
                full_image_path.encode('utf8')),
        'image/format': _bytes_feature(
                file_format.encode('utf8')),
        # 'image/height': _int64_feature(height),
        # 'image/width': _int64_feature(width),
        'image/channels': _int64_feature(3),
        'image/segmentation/filename': _bytes_feature(
                full_label_path.encode('utf8')),
        # 'image/segmentation/class/encoded': _bytes_feature(encoded_label.tostring()),
        'image/segmentation/class/format':_bytes_feature('png'.encode('utf8')),
    }
    
    example = tf.train.Example(
        features=tf.train.Features(feature=feature_dict))
    return example


def _create_tf_record(images, labels, output_path):
    options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    writer = tf.python_io.TFRecordWriter(output_path, options=options)
    if FLAGS.shuffle:
        images, labels = zip(*random.sample(list(zip(images, labels)), len(images)))
    for idx, image in enumerate(images):
        if idx % 100 == 0:
            tf.logging.info('On image %d of %d', idx, len(images))
        tf_example = create_tf_example(
            image, labels[idx], is_jpeg=False)
        writer.write(tf_example.SerializeToString())
    writer.close()
    tf.logging.info('Finished writing!')


def main(_):
    assert FLAGS.output_dir, '`output_dir` missing.'
    assert FLAGS.split_type, '`split_type` missing.'
    assert FLAGS.name, "`name` missing"
    assert (FLAGS.cityscapes_dir) or \
           (FLAGS.input_pattern) or \
           (FLAGS.list_file), \
           'Must specify either `cityscapes_dir` or ' \
           '(`input_pattern` and `annot_pattern`) or `list_file`.'

    if not tf.gfile.IsDirectory(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)
    train_output_path = os.path.join(FLAGS.output_dir,
        '{}_{}.record'.format(FLAGS.name, FLAGS.split_type))

    if FLAGS.cityscapes_dir:
        search_image_files = os.path.join(FLAGS.cityscapes_dir,
            _DEFAULT_DIR['image'], FLAGS.split_type, '*', _DEFAULT_PATTEN['input'])
        search_annot_files = os.path.join(FLAGS.cityscapes_dir,
            _DEFAULT_DIR['label'], FLAGS.split_type, '*', _DEFAULT_PATTEN['label'])
        image_filenames = glob.glob(search_image_files)
        annot_filenames = glob.glob(search_annot_files)
        if len(image_filenames) != len(annot_filenames):
            image_filenames = sorted(image_filenames)
            annot_filenames = sorted(annot_filenames)
            import pdb; pdb.set_trace()
    elif FLAGS.list_file:
        with open(FLAGS.list_file) as f:
            content = f.readlines()
        content = [x.strip().split(",") for x in content]
        image_filenames, annot_filenames = zip(*content)
    else:
        image_filenames = glob.glob(FLAGS.input_pattern)
        if FLAGS.annot_pattern:
            annot_filenames = glob.glob(FLAGS.annot_pattern)
        else:
            print("WARNING: no annotations supplied, using 254")
            annot_filenames = image_filenames.copy()
    
        if len(image_filenames) != len(annot_filenames):
            print("images: ", len(image_filenames))
            print("annot: ", len(annot_filenames))
            
            img_suff = {os.path.splitext(os.path.basename(f))[0]: f for f in image_filenames}
            annot_suff = {os.path.splitext(os.path.basename(f))[0].replace("_train_id", ""): f for f in annot_filenames}
            image_filenames = [img_suff[f] for f in img_suff if f in annot_suff]
            annot_filenames = [annot_suff[f] for f in annot_suff if f in img_suff]
            print("new images:", len(image_filenames))
            print("new annot:", len(annot_filenames))
            #inter = img_suff.intersection(annot_suff)
            #if len(annot_filenames) > len(image_filenames):
            #    print(annot_suff - inter)
            #else:
            #    print(img_suff - inter)

    _create_tf_record(
            sorted(image_filenames),
            sorted(annot_filenames),
            output_path=train_output_path)


if __name__ == '__main__':
    tf.app.run()
