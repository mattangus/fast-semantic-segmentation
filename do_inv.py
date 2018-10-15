import tensorflow as tf
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True)
parser.add_argument("--output", type=str, required=True)


args = parser.parse_args()

mat = np.load(args.input)
if "npz" in args.input:
    mat = mat["arr_0"]

output_shape = mat.shape

new_shape = [-1] + list(mat.shape[-2:])

mat = np.reshape(mat, new_shape)

pl = tf.placeholder(tf.float32, [None] + list(mat.shape[-2:]))

eps = 0.00000001

inv_op = tf.linalg.inv(pl + tf.eye(mat.shape[-1])*eps)

batch = 10000

results = np.zeros(mat.shape)

#import pdb; pdb.set_trace()

with tf.Session() as sess:
    for i in range(int(mat.shape[0]/batch)+1):
        start = i*batch
        end = (i+1)*batch
        if end > mat.shape[0]:
            end = mat.shape[0]
        print(start, end)
        input_batch = mat[start:end]
        print(input_batch.shape)
        inv = sess.run(inv_op, feed_dict={pl: input_batch})
        results[start:end] = inv

    results = np.array(results)
    results = np.reshape(results, output_shape)
    if "npz" in args.output:
        np.savez(args.output, results)
    else:
        np.save(args.output, results)
