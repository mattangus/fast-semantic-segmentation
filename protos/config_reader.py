import tensorflow as tf
from google.protobuf import text_format

from . import pipeline_pb2, input_reader_pb2

def read_config(model_config_file, data_config_file):
    pipeline_config = pipeline_pb2.PipelineConfig()
    with tf.gfile.GFile(model_config_file, 'r') as f:
        text_format.Merge(f.read(), pipeline_config)
    
    input_reader = input_reader_pb2.InputReader()
    with tf.gfile.GFile(data_config_file, 'r') as f:
        text_format.Merge(f.read(), input_reader)
    
    pipeline_config.input_reader.CopyFrom(input_reader)

    return pipeline_config