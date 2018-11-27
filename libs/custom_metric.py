import tensorflow as tf
import numpy as np
from tensorflow.python.ops import metrics_impl

#proofs found here http://people.ds.cam.ac.uk/fanf2/hermes/doc/antiforgery/stats.pdf

def safe_div(a,b):
    b = tf.ones_like(a)*b #broadcast
    return tf.where(tf.less(tf.abs(b), 1e-7), b, a/b)

def streaming_mean(variable, weights=None, has_batch=False):
    """[summary]
    
    streaming mean
    k: iteration 
    x_k: observed data
    w_k: sum of weights after k iters
    m_k: mean after k iters
    Update is as follows:
    m_k = m_{k-1} + (x_k - m_{k-1})w_k/W_k

    Arguments:
        variable {[type]} -- [description]
    
    Keyword Arguments:
        weights {[type]} -- [description] (default: {None})
    
    Raises:
        ValueError -- [description]
        ValueError -- [description]
    
    Returns:
        [type] -- [description]
    """
    static_shape = variable.get_shape().as_list()

    if any([v is None for v in static_shape]):
        raise ValueError("Every dim must be statically defined")

    if weights is not None and len(static_shape) != len(weights.get_shape()):
        raise ValueError("Weights must have the same rank as variable.")

    #weights = tf.expand_dims(weights, -1)
    m_k = metrics_impl.metric_variable(static_shape, tf.float32, name="streaming_mean")
    weight_total = metrics_impl.metric_variable(static_shape, tf.float32, name="weight_total")
    # m_k = tf.get_variable("streaming_mean", initializer=np.zeros(static_shape, dtype=np.float32))
    # weight_total = tf.get_variable("weight_total", initializer=np.zeros(static_shape, dtype=np.float32)) #n_k in formula
    # init = tf.get_variable("init", initializer=np.zeros(static_shape).astype(np.bool))
    
    if weights is not None:
        mask = weights
    else:
        mask = tf.ones(static_shape)

    next_weights = weight_total + mask
    
    #when mask = 1 this is just 1/n
    multiplier = safe_div(weights, next_weights)
    temp = (variable - m_k)*multiplier

    #temp = tf.Print(temp, ["m_k", m_k, "x_k", variable, "n_weight", next_weights, "temp", temp, "weight_t", weight_total, "multiplier", multiplier], summarize=10)

    update_mk = tf.assign(m_k, m_k + temp)
    with tf.control_dependencies([update_mk]):
        update_counts = tf.assign(weight_total, next_weights)

    if has_batch:
        #TODO: axis variable might work here instead of 0
        final_w = safe_div(weight_total, tf.reduce_sum(weight_total,0))
        final_m = tf.reduce_sum(m_k * final_w,0)
    else:
        final_m = tf.identity(m_k)

    return final_m, tf.group(update_mk, update_counts)

def batch_streaming_mean(variable, weights=None, has_batch=False):
    """
    Bach based streaming mean
    k: batch number
    b_k: batch sum
    s_k: sum of batch sizes after batch k
    m_k: mean after k batches
    l_k: size of batck k
    Update is as follows:
    m_k = m_{k-1} + (b_k - m_{k-1}*l_k)/s_k
    """
    static_shape = variable.get_shape().as_list()
    dyn_shape = tf.shape(variable)
    batch = float(1.0)
    start = 0
    if has_batch:
        batch = tf.to_float(dyn_shape[0]) # l_k
        start = 1

    if any([v is None for v in static_shape[start:]]):
        raise ValueError("Every dim except batch must be statically defined")
    
    m_k = tf.get_variable("streaming_mean", initializer=np.zeros(static_shape[start:], dtype=np.float32))
    counts = tf.get_variable("counts", initializer=np.zeros(static_shape[start:], dtype=np.float32)) #s_k in formula
    init = tf.get_variable("init", initializer=np.zeros(static_shape[start:]).astype(np.bool))

    batch_sum = variable
    if batch != float(1.0):
        batch_sum = tf.reduce_sum(variable*weights, 0)
        if weights is not None:
           weights = tf.reduce_sum(weights, 0)
        
    
    if weights is not None:
        temp = (batch_sum - m_k*batch)*weights
        #mask = tf.to_float(tf.not_equal(weights, 0))
        mask = weights
    else:
        temp = (batch_sum - m_k*batch)
        mask = tf.ones(static_shape[1:])
    
    mask_batch = mask*batch
    next_counts = counts + mask_batch

    selected = safe_div(temp, next_counts) #tf.where(init, temp/next_counts, temp)

    selected = tf.Print(selected, ["m_k", m_k, "n_count", next_counts, "temp", temp, "batch_sum", batch_sum, "batch", batch])

    update_mk = tf.assign(m_k, m_k + selected)
    with tf.control_dependencies([update_mk]):
        update_counts = tf.assign(counts, next_counts)
    with tf.control_dependencies([update_counts]):
        update_init = tf.assign(init, tf.logical_or(init, tf.cast(mask,tf.bool)))

    return m_k, tf.group(update_mk, update_counts, update_init)