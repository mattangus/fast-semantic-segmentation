import tensorflow as tf
import numpy as np
from abc import ABC, abstractmethod
from fractions import gcd
from libs import sliding_window
from libs.custom_metric import streaming_mean
from submod.cholesky.cholesky_update import cholesky_update, _metric_variable

PATCH_SIZE = [3, 3]

class StatComputer(ABC):
    
    @abstractmethod
    def get_update_op(self):
        pass
    
    @abstractmethod
    def get_reset_op(self):
        pass

    @abstractmethod
    def save_variable(self, sess, stat_dir):
        pass
    
    @abstractmethod
    def get_variable(self, sess, stat_dir):
        pass

class MeanComputer(StatComputer):

    def __init__(self, values, weights):
        self.values = values
        self.weights = weights

        with tf.variable_scope("MeanComputer"):
            (self.mean, self.mean_ref), self.update = streaming_mean(self.values, self.weights, True)
            self.mean = tf.expand_dims(self.mean,0)
    
    def get_update_op(self):
        return self.update
    
    def get_reset_op(self):
        return tf.assign(self.mean_ref, tf.zeros_like(self.mean_ref))

    def get_variable(self, sess):
        mean_value = sess.run(self.mean)
        if np.isnan(mean_value).any():
            print("nan time")
            import pdb; pdb.set_trace()
        return mean_value

    def save_variable(self, sess, file_name):
        mean_value = self.get_variable(sess)
        print("saving to", file_name)
        np.savez(file_name, mean_value)

class CovComputer(StatComputer):

    def __init__(self, values, mask, mean):
        self.values = values
        self.mask = mask
        self.mean = mean

        self.mean_sub = values - mean
        self.weighted_values = tf.sqrt(mask) * self.mean_sub
        self.batch_values = tf.reshape(self.weighted_values, [-1, tf.shape(values)[-1]])
        self.batch_mask = tf.reshape(mask, [-1])
        
        self.count = _metric_variable(self.batch_mask.shape, tf.float32, initializer=tf.ones(self.batch_mask.shape))
        self.count_update = tf.assign_add(self.count, self.batch_mask)

        self.chol, self.chol_update = cholesky_update(self.batch_values, self.batch_mask, init=float(1.0))
    
    def get_update_op(self):
        return tf.group([self.chol_update, self.count_update])
    
    def get_reset_op(self):
        return tf.group([tf.variables_initializer(self.chol), tf.variables_initializer(self.count)])

    def get_variable(self, sess):
        def inv_fn(chol_mat, counts):
            cov = tf.matmul(tf.transpose(chol_mat,[0,2,1]),chol_mat)
            #cov = cov / tf.expand_dims(tf.expand_dims(counts,-1),-1)
            inv_cov = tf.linalg.inv(cov) * tf.expand_dims(tf.expand_dims(counts,-1),-1)
            return inv_cov
        target_shape = self.values.get_shape().as_list()
        in_size = self.chol.get_shape().as_list()[0]
        to_check = [(a, in_size) for a in range(1,21)]
        num_split = max(map(lambda v: gcd(v[0],v[1]), to_check))
        print("using split", num_split)
        
        chol_list = tf.split(self.chol, num_split, 0)
        count_list = tf.split(self.count, num_split, 0)
        inv_list = [inv_fn(ch, co) for ch,co in zip(chol_list,count_list)]
        class_cov_inv = np.concatenate([sess.run(i) for i in inv_list])
        #TODO: Add weight divisor
        class_cov_inv = np.sum(np.reshape(class_cov_inv, target_shape + [target_shape[-1]]), 0, keepdims=True)
        
        if np.isnan(class_cov_inv).any():
            print("nan time")
            import pdb; pdb.set_trace()
        
        return class_cov_inv
    
    def save_variable(self, sess, file_name):
        class_cov_inv = self.get_variable(sess)
        print("saving to", file_name)
        np.savez(file_name, class_cov_inv)

def get_patches(values, patch_shape):
    assert len(patch_shape) == 2, "patch_shape must have length 2"
    assert patch_shape[0] == patch_shape[1], "Only square shape supported"
    temp_shape = values.shape.as_list()
    assert np.sum([t is None for t in temp_shape]) <= 1, "Can only have one None shape"

    temp_shape = [-1 if t is None else t for t in temp_shape]
    temp = tf.reshape(values, temp_shape[:3] + [np.prod(temp_shape[3:])])

    patches = sliding_window.extract_patches(temp, patch_shape[0], patch_shape[1], patch_shape[1])
    patches = tf.reshape(patches, [-1] + patch_shape + temp_shape[3:])
    return patches

class PatchMeanComputer(StatComputer):

    def __init__(self, values, weights):
        self.values = values
        self.weights = weights

        with tf.variable_scope("MeanComputer"):
            self.values = get_patches(self.values, PATCH_SIZE)
            self.weights = get_patches(self.weights, PATCH_SIZE)
            self.mean, self.update = streaming_mean(self.values, self.weights, True)
            self.mean = tf.expand_dims(self.mean,0)

    def get_update_op(self):
        return self.update
    
    def save_variable(self, sess, file_name):
        mean_value = sess.run(self.mean)
        if np.isnan(mean_value).any():
            print("nan time")
            import pdb; pdb.set_trace()
        print("saving to", file_name)
        np.savez(file_name, mean_value)

class PatchCovComputer(StatComputer):

    def __init__(self, values, mask, mean):
        self.values = values
        self.mask = mask
        self.mean = mean
        
        self.values = temp_val = get_patches(self.values, PATCH_SIZE)
        self.mask = get_patches(self.mask, PATCH_SIZE)     

        self.mean_sub = self.values - mean
        self.batch_values = tf.reshape(self.mean_sub, [-1, tf.shape(self.values)[-1]])
        self.batch_mask = tf.reshape(self.mask, [-1])
        self.chol, self.chol_update = cholesky_update(self.batch_values, self.batch_mask, init=float(1.0))
    
    def get_update_op(self):
        return self.chol_update
    
    def save_variable(self, sess, file_name):
        def inv_fn(chol_mat):
            cov = tf.matmul(tf.transpose(chol_mat,[0,2,1]),chol_mat)
            inv_cov = tf.linalg.inv(cov)
            return inv_cov
        target_shape = self.values.get_shape().as_list()
        in_size = self.chol.get_shape().as_list()[0]
        to_check = [(a, in_size) for a in range(1,21)]
        num_split = max(map(lambda v: gcd(v[0],v[1]), to_check))
        print("using split", num_split)
        chol_list = tf.split(self.chol, num_split, 0)
        inv_list = [inv_fn(c) for c in chol_list]
        class_cov_inv = np.concatenate([sess.run(i) for i in inv_list])
        class_cov_inv = np.mean(np.reshape(class_cov_inv, target_shape + [target_shape[-1]]), 0, keepdims=True)

        if np.isnan(class_cov_inv).any():
            print("nan time")
            import pdb; pdb.set_trace()
        print("saving to", file_name)
        np.savez(file_name, class_cov_inv)
