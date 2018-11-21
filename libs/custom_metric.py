import tensorflow as tf
import numpy as np

def safe_div(a,b):
    b = tf.ones_like(a)*b #broadcast
    return tf.where(tf.less(tf.abs(b), 1e-7), b, a/b)

def streaming_mean(variable, weights=None, has_batch=False):
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