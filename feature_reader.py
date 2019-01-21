import numpy as np
import tensorflow as tf
import json
import glob

def example_dict(name):
    ret_dict = {
        name + "/data": tf.FixedLenFeature([], tf.string),
        # name + "/shape": tf.FixedLenFeature([], tf.int64),
        # name + "/dtype": tf.FixedLenFeature([], tf.string)
    }
    return ret_dict

def example_to_tensor(parsed, name, config):
    decoded = tf.decode_raw(parsed[name + "/data"], config[name]["dtype"])
    decoded = tf.reshape(decoded, config[name]["shape"])

    return decoded

def _decode_example(proto, config):

    feature_dict = {
        "filename": tf.FixedLenFeature([], tf.string)
    }

    names = ["input", "annot", "pred", "final_logits", "unscaled_logits"]

    for name in names:
        feature_dict.update(example_dict(name))

    parsed = tf.parse_single_example(proto, feature_dict)

    ret = {}
    for name in names:
        ret[name] = example_to_tensor(parsed, name, config)

    return ret

def get_feature_dataset(record_prefix, batch):
    filenames = glob.glob(record_prefix + "-*")
    with open(record_prefix + ".config", "r") as f:
        config = json.load(f)
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(lambda proto: _decode_example(proto, config))
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(32)

    return dataset