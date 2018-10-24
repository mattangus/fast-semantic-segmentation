import tensorflow as tf

def extract_patches(inputs, crop_height, crop_width):
    c = inputs.get_shape()[-1]
    ksizes = [1, crop_height, crop_width, 1]
    strides = [1, crop_height//3, crop_width//3, 1]
    rates = [1,1,1,1]
    padding = "SAME"
    patches = tf.extract_image_patches(inputs, ksizes=ksizes, strides=strides, rates=rates, padding=padding)
    patches_shape = tf.shape(patches)
    return patches #tf.reshape(patches, [tf.reduce_prod(patches_shape[0:3]), crop_height, crop_width, int(c)])

def merge_patches(x, y, crop_height, crop_width):
    _x = tf.zeros_like(x)
    _y = extract_patches(_x, crop_height, crop_width)
    grad = tf.gradients(_y, _x)[0]
    # Divide by grad, to "average" together the overlapping patches
    # otherwise they would simply sum up
    return tf.gradients(_y, _x, grad_ys=y)[0] / grad