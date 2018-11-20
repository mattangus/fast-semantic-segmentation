from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import sets
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.ops import metrics_impl
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import distribute as distribute_lib
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export

def _safe_div(numerator, denominator, name):
  """Divides two values, returning 0 if the denominator is <= 0.
  Args:
    numerator: A real `Tensor`.
    denominator: A real `Tensor`, with dtype matching `numerator`.
    name: Name for the returned op.
  Returns:
    0 if `denominator` <= 0, else `numerator` / `denominator`
  """
  return array_ops.where(
      math_ops.greater(denominator, 0),
      math_ops.truediv(numerator, denominator),
      0,
      name=name)

def metric_variable(shape, dtype, validate_shape=True, name=None):
  """Create variable in `GraphKeys.(LOCAL|METRIC_VARIABLES)` collections.
  If running in a `DistributionStrategy` context, the variable will be
  "tower local". This means:
  *   The returned object will be a container with separate variables
      per replica/tower of the model.
  *   When writing to the variable, e.g. using `assign_add` in a metric
      update, the update will be applied to the variable local to the
      replica/tower.
  *   To get a metric's result value, we need to sum the variable values
      across the replicas/towers before computing the final answer.
      Furthermore, the final answer should be computed once instead of
      in every replica/tower. Both of these are accomplished by
      running the computation of the final result value inside
      `tf.contrib.distribute.get_tower_context().merge_call(fn)`.
      Inside the `merge_call()`, ops are only added to the graph once
      and access to a tower-local variable in a computation returns
      the sum across all replicas/towers.
  Args:
    shape: Shape of the created variable.
    dtype: Type of the created variable.
    validate_shape: (Optional) Whether shape validation is enabled for
      the created variable.
    name: (Optional) String name of the created variable.
  Returns:
    A (non-trainable) variable initialized to zero, or if inside a
    `DistributionStrategy` scope a tower-local variable container.
  """
  # Note that synchronization "ON_READ" implies trainable=False.
  return variable_scope.variable(
      lambda: array_ops.zeros(shape, dtype),
      collections=[
          ops.GraphKeys.LOCAL_VARIABLES, ops.GraphKeys.METRIC_VARIABLES
      ],
      validate_shape=validate_shape,
      #synchronization=3, #on read
      #aggregation=1, #sum
      name=name)


def _streaming_confusion_matrix(labels, predictions, num_classes, weights=None):
  """Calculate a streaming confusion matrix.
  Calculates a confusion matrix. For estimation over a stream of data,
  the function creates an  `update_op` operation.
  Args:
    labels: A `Tensor` of ground truth labels with shape [batch size] and of
      type `int32` or `int64`. The tensor will be flattened if its rank > 1.
    predictions: A `Tensor` of prediction results for semantic labels, whose
      shape is [batch size] and type `int32` or `int64`. The tensor will be
      flattened if its rank > 1.
    num_classes: The possible number of labels the prediction task can
      have. This value must be provided, since a confusion matrix of
      dimension = [num_classes, num_classes] will be allocated.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `labels` dimension).
  Returns:
    total_cm: A `Tensor` representing the confusion matrix.
    update_op: An operation that increments the confusion matrix.
  """
  # Local variable to accumulate the predictions in the confusion matrix.
  total_cm = metric_variable(
      [num_classes, num_classes], dtypes.float64, name='total_confusion_matrix')

  # Cast the type to int64 required by confusion_matrix_ops.
  predictions = math_ops.to_int64(predictions)
  labels = math_ops.to_int64(labels)
  num_classes = math_ops.to_int64(num_classes)

  # Flatten the input if its rank > 1.
  if predictions.get_shape().ndims > 1:
    predictions = array_ops.reshape(predictions, [-1])

  if labels.get_shape().ndims > 1:
    labels = array_ops.reshape(labels, [-1])

  if (weights is not None) and (weights.get_shape().ndims > 1):
    weights = array_ops.reshape(weights, [-1])

  # Accumulate the prediction to current confusion matrix.
  current_cm = confusion_matrix.confusion_matrix(
      labels, predictions, num_classes, weights=weights, dtype=dtypes.float64)
  update_op = state_ops.assign_add(total_cm, current_cm)
  return total_cm, update_op

def mean_iou(labels,
             predictions,
             num_classes,
             weights=None,
             metrics_collections=None,
             updates_collections=None,
             name=None):
  """Calculate per-step mean Intersection-Over-Union (mIOU).
  Mean Intersection-Over-Union is a common evaluation metric for
  semantic image segmentation, which first computes the IOU for each
  semantic class and then computes the average over classes.
  IOU is defined as follows:
    IOU = true_positive / (true_positive + false_positive + false_negative).
  The predictions are accumulated in a confusion matrix, weighted by `weights`,
  and mIOU is then calculated from it.
  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates these variables and returns the `mean_iou`.
  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.
  Args:
    labels: A `Tensor` of ground truth labels with shape [batch size] and of
      type `int32` or `int64`. The tensor will be flattened if its rank > 1.
    predictions: A `Tensor` of prediction results for semantic labels, whose
      shape is [batch size] and type `int32` or `int64`. The tensor will be
      flattened if its rank > 1.
    num_classes: The possible number of labels the prediction task can
      have. This value must be provided, since a confusion matrix of
      dimension = [num_classes, num_classes] will be allocated.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `labels` dimension).
    metrics_collections: An optional list of collections that `mean_iou`
      should be added to.
    updates_collections: An optional list of collections `update_op` should be
      added to.
    name: An optional variable_scope name.
  Returns:
    mean_iou: A `Tensor` representing the mean intersection-over-union.
    update_op: An operation that increments the confusion matrix.
  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, or if
      `weights` is not `None` and its shape doesn't match `predictions`, or if
      either `metrics_collections` or `updates_collections` are not a list or
      tuple.
    RuntimeError: If eager execution is enabled.
  """
  if context.executing_eagerly():
    raise RuntimeError('tf.metrics.mean_iou is not supported when '
                       'eager execution is enabled.')

  with variable_scope.variable_scope(name, 'mean_iou',
                                     (predictions, labels, weights)):
    # Check if shape is compatible.
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())

    total_cm, update_op = _streaming_confusion_matrix(labels, predictions,
                                                      num_classes, weights)

    def compute_mean_iou(total_cm, name):
      """Compute the mean intersection-over-union via the confusion matrix."""
      sum_over_row = math_ops.to_float(math_ops.reduce_sum(total_cm, 0))
      sum_over_col = math_ops.to_float(math_ops.reduce_sum(total_cm, 1))
      cm_diag = math_ops.to_float(array_ops.diag_part(total_cm))
      denominator = sum_over_row + sum_over_col - cm_diag

      # The mean is only computed over classes that appear in the
      # label or prediction tensor. If the denominator is 0, we need to
      # ignore the class.
      num_valid_entries = math_ops.reduce_sum(
          math_ops.cast(
              math_ops.not_equal(denominator, 0), dtype=dtypes.float32))

      # If the value of the denominator is 0, set it to 1 to avoid
      # zero division.
      denominator = array_ops.where(
          math_ops.greater(denominator, 0), denominator,
          array_ops.ones_like(denominator))
      iou = math_ops.div(cm_diag, denominator)

      # If the number of valid entries is 0 (no classes) we return 0.
      result = array_ops.where(
          math_ops.greater(num_valid_entries, 0),
          math_ops.reduce_sum(iou, name=name) / num_valid_entries, 0)
      return result

    def mean_iou_across_towers(_, v):
      mean_iou_v = compute_mean_iou(v, 'mean_iou')
      if metrics_collections:
        ops.add_to_collections(metrics_collections, mean_iou_v)
      return mean_iou_v

    mean_iou_v = distribute_lib.get_tower_context().merge_call(
        mean_iou_across_towers, total_cm)

    if updates_collections:
      ops.add_to_collections(updates_collections, update_op)

    return mean_iou_v, total_cm, update_op

def streaming_covariance(predictions,
                         labels,
                         weights=None,
                         metrics_collections=None,
                         updates_collections=None,
                         name=None):
  """Computes the unbiased sample covariance between `predictions` and `labels`.
  The `streaming_covariance` function creates four local variables,
  `comoment`, `mean_prediction`, `mean_label`, and `count`, which are used to
  compute the sample covariance between predictions and labels across multiple
  batches of data. The covariance is ultimately returned as an idempotent
  operation that simply divides `comoment` by `count` - 1. We use `count` - 1
  in order to get an unbiased estimate.
  The algorithm used for this online computation is described in
  https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance.
  Specifically, the formula used to combine two sample comoments is
  `C_AB = C_A + C_B + (E[x_A] - E[x_B]) * (E[y_A] - E[y_B]) * n_A * n_B / n_AB`
  The comoment for a single batch of data is simply
  `sum((x - E[x]) * (y - E[y]))`, optionally weighted.
  If `weights` is not None, then it is used to compute weighted comoments,
  means, and count. NOTE: these weights are treated as "frequency weights", as
  opposed to "reliability weights". See discussion of the difference on
  https://wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_variance
  To facilitate the computation of covariance across multiple batches of data,
  the function creates an `update_op` operation, which updates underlying
  variables and returns the updated covariance.
  Args:
    predictions: A `Tensor` of arbitrary size.
    labels: A `Tensor` of the same size as `predictions`.
    weights: Optional `Tensor` indicating the frequency with which an example is
      sampled. Rank must be 0, or the same rank as `labels`, and must be
      broadcastable to `labels` (i.e., all dimensions must be either `1`, or
      the same as the corresponding `labels` dimension).
    metrics_collections: An optional list of collections that the metric
      value variable should be added to.
    updates_collections: An optional list of collections that the metric update
      ops should be added to.
    name: An optional variable_scope name.
  Returns:
    covariance: A `Tensor` representing the current unbiased sample covariance,
      `comoment` / (`count` - 1).
    update_op: An operation that updates the local variables appropriately.
  Raises:
    ValueError: If labels and predictions are of different sizes or if either
      `metrics_collections` or `updates_collections` are not a list or tuple.
  """
  with variable_scope.variable_scope(name, 'covariance',
                                     (predictions, labels, weights)):
    predictions, labels, weights = metrics_impl._remove_squeezable_dimensions(  # pylint: disable=protected-access
        predictions, labels, weights)
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())
    count_ = metric_variable([], dtypes.float32, name='count')
    mean_prediction = metric_variable(
        [], dtypes.float32, name='mean_prediction')
    mean_label = metric_variable(
        [], dtypes.float32, name='mean_label')
    comoment = metric_variable(  # C_A in update equation
        [], dtypes.float32, name='comoment')

    if weights is None:
      batch_count = math_ops.to_float(array_ops.size(labels))  # n_B in eqn
      weighted_predictions = predictions
      weighted_labels = labels
    else:
      weights = weights_broadcast_ops.broadcast_weights(weights, labels)
      batch_count = math_ops.reduce_sum(weights)  # n_B in eqn
      weighted_predictions = math_ops.multiply(predictions, weights)
      weighted_labels = math_ops.multiply(labels, weights)

    update_count = state_ops.assign_add(count_, batch_count)  # n_AB in eqn
    prev_count = update_count - batch_count  # n_A in update equation

    # We update the means by Delta=Error*BatchCount/(BatchCount+PrevCount)
    # batch_mean_prediction is E[x_B] in the update equation
    batch_mean_prediction = _safe_div(
        math_ops.reduce_sum(weighted_predictions), batch_count,
        'batch_mean_prediction')
    delta_mean_prediction = _safe_div(
        (batch_mean_prediction - mean_prediction) * batch_count, update_count,
        'delta_mean_prediction')
    update_mean_prediction = state_ops.assign_add(mean_prediction,
                                                  delta_mean_prediction)
    # prev_mean_prediction is E[x_A] in the update equation
    prev_mean_prediction = update_mean_prediction - delta_mean_prediction

    # batch_mean_label is E[y_B] in the update equation
    batch_mean_label = _safe_div(
        math_ops.reduce_sum(weighted_labels), batch_count, 'batch_mean_label')
    delta_mean_label = _safe_div((batch_mean_label - mean_label) * batch_count,
                                 update_count, 'delta_mean_label')
    update_mean_label = state_ops.assign_add(mean_label, delta_mean_label)
    # prev_mean_label is E[y_A] in the update equation
    prev_mean_label = update_mean_label - delta_mean_label

    unweighted_batch_coresiduals = ((predictions - batch_mean_prediction) *
                                    (labels - batch_mean_label))
    # batch_comoment is C_B in the update equation
    if weights is None:
      batch_comoment = math_ops.reduce_sum(unweighted_batch_coresiduals)
    else:
      batch_comoment = math_ops.reduce_sum(
          unweighted_batch_coresiduals * weights)

    # View delta_comoment as = C_AB - C_A in the update equation above.
    # Since C_A is stored in a var, by how much do we need to increment that var
    # to make the var = C_AB?
    delta_comoment = (
        batch_comoment + (prev_mean_prediction - batch_mean_prediction) *
        (prev_mean_label - batch_mean_label) *
        (prev_count * batch_count / update_count))
    update_comoment = state_ops.assign_add(comoment, delta_comoment)

    covariance = array_ops.where(
        math_ops.less_equal(count_, 1.),
        float('nan'),
        math_ops.truediv(comoment, count_ - 1),
        name='covariance')
    with ops.control_dependencies([update_comoment]):
      update_op = array_ops.where(
          math_ops.less_equal(count_, 1.),
          float('nan'),
          math_ops.truediv(comoment, count_ - 1),
          name='update_op')

  if metrics_collections:
    ops.add_to_collections(metrics_collections, covariance)

  if updates_collections:
    ops.add_to_collections(updates_collections, update_op)

  return covariance, update_op

def streaming_mean(variable, weights=None):
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
    if len(static_shape) == 4:
        batch = tf.to_float(dyn_shape[0]) # l_k
    if any([v is None for v in static_shape[1:]]):
        raise ValueError("Every dim except batch must be statically defined")
    
    m_k = tf.get_variable("streaming_mean", initializer=np.zeros(static_shape[1:], dtype=np.float32))
    counts = tf.get_variable("counts", initializer=np.zeros(static_shape[1:], dtype=np.float32)) #s_k in formula
    init = tf.get_variable("init", initializer=np.zeros(static_shape[1:]).astype(np.bool))

    batch_sum = variable
    if batch != float(1.0):
        batch_sum = tf.reduce_sum(variable, 0)
    
    if weights:
        temp = (batch_sum - m_k*batch)*weights
        mask = tf.to_float(tf.not_equal(weights, 0))
    else:
        temp = (batch_sum - m_k*batch)
        mask = tf.ones(static_shape[1:])
    
    mask_batch = mask*batch
    next_counts = counts + mask_batch

    selected = tf.where(init, temp, temp)

    update_mk = tf.assign(m_k, m_k + selected/next_counts)
    with tf.control_dependencies([update_mk]):
        update_counts = tf.assign(counts, next_counts)
    with tf.control_dependencies([update_counts]):
        update_init = tf.assign(init, tf.logical_or(init, tf.cast(mask,tf.bool)))

    return m_k, tf.group(update_mk, update_counts, update_init)