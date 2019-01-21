from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from builders import dataset_builder
from builders import preprocessor_builder as preprocessor


slim = tf.contrib.slim


def _map_to_colored_labels(segmentation_map, shape_list, color_map):
    # resolve shapes
    num_classes = len(color_map)
    output_channels = len(color_map[0])
    # convert label map format
    color_map_constant_mat = []
    for color in color_map:
        color_map_constant_mat.append(list(color))
    color_table = tf.constant(color_map_constant_mat, dtype=tf.float32)
    segmentation_map = tf.cast(segmentation_map, dtype=tf.int32)
    onehot_labels = tf.one_hot(segmentation_map, depth=num_classes)
    onehot_labels = tf.reshape(onehot_labels, (-1, num_classes))
    colored_label = tf.matmul(onehot_labels, color_table)
    colored_label = tf.reshape(colored_label,
        (-1, shape_list[1], shape_list[2], output_channels))
    return colored_label

def _get_outputs_from_inputs(model, input_tensors,
                             output_collection_name):
    # models expect a batch dimension
    if len(input_tensors.get_shape()) < 4:
        input_tensors = tf.expand_dims(input_tensors, axis=0)
    # build model, which expects a floating point input
    inputs = tf.to_float(input_tensors)
    preprocessed_inputs = model.preprocess(inputs)
    outputs_dict = model.predict(preprocessed_inputs)
    output_tensors = outputs_dict[model.main_class_predictions_key]
    prediction_tensor = tf.argmax(output_tensors, 3)
    prediction_tensor = tf.expand_dims(prediction_tensor, -1)
    # name tensor to make inference with frozen weights easier
    outputs_dict[model.main_class_predictions_key] = tf.identity(prediction_tensor,
        name=output_collection_name)
    return outputs_dict


def _image_tensor_input_placeholder(input, input_shape=None, pad_to_shape=None, input_type=tf.uint8):
    if input_shape is None:
        input_shape = (None, None, None, 3)
    if input is None:
        placeholder_tensor = tf.placeholder(
            dtype=input_type, shape=input_shape, name='inputs')
    else:
        placeholder_tensor = input
    if pad_to_shape is not None:
        input_tensor = tf.image.pad_to_bounding_box(placeholder_tensor,
            0, 0, pad_to_shape[0], pad_to_shape[1])
    else:
        input_tensor = placeholder_tensor
    return placeholder_tensor, input_tensor


def deploy_segmentation_inference_graph(model, input_shape,
                                        input=None,
                                        pad_to_shape=None,
                                        label_color_map=None,
                                        input_type=tf.uint8,
                                        output_collection_name="predictions"):
    (placeholder_tensor,
      input_tensor) = _image_tensor_input_placeholder(input, input_shape, pad_to_shape, input_type)
    outputs = _get_outputs_from_inputs(model, input_tensor,
            output_collection_name=output_collection_name)
    predictions = outputs[model.main_class_predictions_key]
    if label_color_map is not None:
        output_shape = predictions.get_shape().as_list()
        predictions = _map_to_colored_labels(predictions, output_shape, label_color_map)

    if pad_to_shape is not None:
        if len(input_shape) < 4:
            height = input_shape[0] #no batch
            width = input_shape[1]
        else:
            height = input_shape[1] #has batch
            width = input_shape[2]
        predictions = tf.image.crop_to_bounding_box(predictions, 0, 0, height, width)
    # if model.final_logits_key in outputs:
    #     logits = outputs[model.final_logits_key]
    #     logits = tf.image.crop_to_bounding_box(
    #         logits, 0, 0, input_shape[0], input_shape[1])
    #     outputs[model.final_logits_key] = logits

    tf.train.get_or_create_global_step()
    outputs[model.main_class_predictions_key] = predictions
    return outputs, placeholder_tensor
