from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import time
import tensorflow as tf
import numpy as np

from third_party import model_deploy
from third_party import mem_util

from builders import model_builder
from builders import dist_builder
from builders import dataset_builder
from builders import preprocessor_builder
from builders import optimizer_builder

import queue

from tensorflow.contrib import graph_editor as ge
from tensorflow.contrib.memory_stats.python.ops.memory_stats_ops import BytesInUse

slim = tf.contrib.slim

prefetch_queue = slim.prefetch_queue


def create_training_input(create_input_fn,
                          preprocess_fn,
                          batch_size,
                          batch_queue_capacity,
                          batch_queue_threads,
                          prefetch_queue_capacity):

    tensor_dict = create_input_fn()
    
    def cast_and_reshape(tensor_dict, dicy_key):
        items = tensor_dict[dicy_key]
        float_images = tf.to_float(items)
        tensor_dict[dicy_key] = float_images
        return tensor_dict

    tensor_dict = cast_and_reshape(tensor_dict,
                    dataset_builder._IMAGE_FIELD)

    if preprocess_fn is not None:
        preprocessor = preprocess_fn()
        tensor_dict = preprocessor(tensor_dict)

    batched_tensors = tf.train.batch(tensor_dict,
        batch_size=batch_size, num_threads=batch_queue_threads,
        capacity=batch_queue_capacity, dynamic_pad=True)

    return prefetch_queue.prefetch_queue(batched_tensors,
        capacity=prefetch_queue_capacity,
        dynamic_pad=False)


def create_training_model_losses(input_queue, create_model_fn, train_config,
                                 train_dir=None, gradient_checkpoints=None, clone_scope=None):

    tf.logging.info('Creating clone %s', clone_scope)
    _, segmentation_model = create_model_fn()

    # Optional quantization
    if train_config.quantize_with_delay:
        tf.logging.info('Adding quantization nodes to training graph...')
        tf.contrib.quantize.create_training_graph(
            quant_delay=train_config.quantize_with_delay)

    read_data_list = input_queue.dequeue()
    def extract_images_and_targets(read_data):
        images = read_data[dataset_builder._IMAGE_FIELD]
        labels = read_data[dataset_builder._LABEL_FIELD]
        return (images, labels)

    (images, labels) = zip(*map(extract_images_and_targets, [read_data_list]))
    
    #labels = tf.Print(labels, ["labels max", tf.reduce_max(labels)])

    # Incase we need to do zero centering, we do it here
    preprocessed_images = []
    for image in images:
        resized_image = segmentation_model.preprocess(image)
        preprocessed_images.append(resized_image)
    images = tf.concat(preprocessed_images, 0, name="Inputs")

    segmentation_model.provide_groundtruth(labels[0])
    prediction_dict = segmentation_model.predict(images)

    # Add checkpointing nodes to correct collection
    if gradient_checkpoints is not None:
        tf.logging.info(
            'Adding gradient checkpoints to `checkpoints` collection')
        graph = tf.get_default_graph()
        checkpoint_list = gradient_checkpoints
        for checkpoint_node_name in checkpoint_list:
            curr_tensor_name = clone_scope + checkpoint_node_name + ":0"
            node = graph.get_tensor_by_name(curr_tensor_name)
            tf.add_to_collection('checkpoints', node)

    # Gather main and aux losses here to single collection
    losses_dict = segmentation_model.loss(prediction_dict)
    for loss_tensor in losses_dict.values():
        tf.losses.add_loss(loss_tensor)

def do_back(op):
    result = list(seed_ops)
    wave = set(seed_ops)
    while wave:
        new_wave = set()
        for op in wave:
            for new_t in op.inputs:
                if new_t in stop_at_ts:
                    continue
                if new_t.op not in result and is_within(new_t.op):
                    new_wave.add(new_t.op)
        util.concatenate_unique(result, new_wave)
        wave = new_wave
    
    # q = queue.Queue()
    # q.put(ge.sgv(op))

    # import pdb; pdb.set_trace()
    # res = []

    # while not q.empty():
    #     cur = q.get_nowait()
        
    #     ins = list(cur.inputs)

    #     for it in ins:
    #         q.put_nowait(ge.sgv(it))
    #         res.append(it)
    
    import pdb; pdb.set_trace()
    print("done")

def my_add_check_numerics_ops():
    """Connect a `check_numerics` to every floating point tensor.
    `check_numerics` operations themselves are added for each `half`, `float`,
    or `double` tensor in the graph. For all ops in the graph, the
    `check_numerics` op for all of its (`half`, `float`, or `double`) inputs
    is guaranteed to run before the `check_numerics` op on any of its outputs.
    Note: This API is not compatible with the use of `tf.cond` or
    `tf.while_loop`, and will raise a `ValueError` if you attempt to call it
    in such a graph.
    Returns:
        A `group` op depending on all `check_numerics` ops added.
    """
    from tensorflow.python.framework import ops
    from tensorflow.python.ops import control_flow_ops
    check_op = []
    # This code relies on the ordering of ops in get_operations().
    # The producer of a tensor always comes before that tensor's consumer in
    # this list. This is true because get_operations() returns ops in the order
    # added, and an op can only be added after its inputs are added.
    for op in ops.get_default_graph().get_operations():
        for output in op.outputs:
            if output.dtype in [tf.float16, tf.float32, tf.float64]:
                message = op.name + ":" + str(output.value_index)
                if op._get_control_flow_context() is not None:  # pylint: disable=protected-access
                    print("not adding numerics check for", message)
                else:
                    with ops.control_dependencies(check_op):
                        check_op = [tf.check_numerics(output, message=message)]
    return control_flow_ops.group(*check_op)

def train_segmentation_model(create_model_fn,
                             create_input_fn,
                             train_config,
                             model_config,
                             master,
                             task,
                             is_chief,
                             startup_delay_steps,
                             train_dir,
                             num_clones,
                             num_worker_replicas,
                             num_ps_tasks,
                             clone_on_cpu,
                             replica_id,
                             num_replicas,
                             max_checkpoints_to_keep,
                             save_interval_secs,
                             image_summaries,
                             log_memory=False,
                             gradient_checkpoints=None,
                             sync_bn_accross_gpu=False):
    """Create an instance of the FastSegmentationModel"""
    _, segmentation_model = create_model_fn()
    deploy_config = model_deploy.DeploymentConfig(
        num_clones=num_clones,
        clone_on_cpu=clone_on_cpu,
        replica_id=task,
        num_replicas=num_worker_replicas,
        num_ps_tasks=num_ps_tasks)
    startup_delay_steps = task * startup_delay_steps

    per_clone_batch_size = train_config.batch_size #// num_clones

    preprocess_fn = None
    if train_config.preprocessor_step:
        preprocess_fn = functools.partial(
            preprocessor_builder.build,
            preprocessor_config_list=train_config.preprocessor_step)

    with tf.Graph().as_default():
        # CPU of common ps server
        with tf.device(deploy_config.variables_device()):
            global_step = tf.train.get_or_create_global_step()

        with tf.device(deploy_config.inputs_device()): # CPU of each worker
            input_queue = create_training_input(
                create_input_fn,
                preprocess_fn,
                per_clone_batch_size,
                batch_queue_capacity=train_config.batch_queue_capacity,
                batch_queue_threads=train_config.num_batch_queue_threads,
                prefetch_queue_capacity=train_config.prefetch_queue_capacity)

        # Create the global step on the device storing the variables.
        with tf.device(deploy_config.variables_device()):
            # Note: it is assumed that any loss created by `model_fn`
            # is collected at the tf.GraphKeys.LOSSES collection.
            model_fn = functools.partial(create_training_model_losses,
                                    create_model_fn=create_model_fn,
                                    train_config=train_config,
                                    train_dir=train_dir,
                                    gradient_checkpoints=gradient_checkpoints)
            clones = model_deploy.create_clones(deploy_config,
                model_fn, [input_queue])
            first_clone_scope = deploy_config.clone_scope(0)

            if sync_bn_accross_gpu:
                # Attempt to sync BN updates across all GPU's in a tower.
                # Caution since this is very slow. Might not be needed
                update_ops = []
                for idx in range(num_clones):
                    nth_clone_sope = deploy_config.clone_scope(0)
                    update_ops.extend(tf.get_collection(
                        tf.GraphKeys.UPDATE_OPS, nth_clone_sope))
            else:
                # Gather updates from first GPU only
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,
                                               first_clone_scope)

        # Init variable to collect summeries
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        # Add summaries for losses.
        for loss in tf.get_collection(tf.GraphKeys.LOSSES, first_clone_scope):
            summaries.add(tf.summary.scalar('Losses/%s' % loss.op.name, loss))

        with tf.device(deploy_config.optimizer_device()): # CPU of each worker
            (training_optimizer,
              optimizer_summary_vars) = optimizer_builder.build(
                train_config.optimizer, num_clones)
            for var in optimizer_summary_vars:
                summaries.add(
                    tf.summary.scalar(var.op.name, var, family='LearningRate'))

        # Add summaries for model variables.
        # for model_var in slim.get_model_variables():
        #     summaries.add(tf.summary.histogram(model_var.op.name, model_var))

        # Fine tune from classification or segmentation checkpoints
        trainable_vars = tf.get_collection(
                              tf.GraphKeys.TRAINABLE_VARIABLES)
        if train_config.fine_tune_checkpoint:
            if not train_config.fine_tune_checkpoint_type:
                raise ValueError('Must specify `fine_tune_checkpoint_type`.')

            tf.logging.info('Initializing %s model from checkpoint %s',
                train_config.fine_tune_checkpoint_type,
                train_config.fine_tune_checkpoint)

            variables_to_restore = segmentation_model.restore_map(train_config.fine_tune_checkpoint,
              fine_tune_checkpoint_type=train_config.fine_tune_checkpoint_type)

            init_fn = slim.assign_from_checkpoint_fn(
                        train_config.fine_tune_checkpoint,
                        variables_to_restore,
                        ignore_missing_vars=True)

            if train_config.freeze_fine_tune_backbone:
                tf.logging.info('Freezing %s scope from checkpoint.')
                non_frozen_vars = []
                for var in trainable_vars:
                    if not var.op.name.startswith(
                      segmentation_model.shared_feature_extractor_scope):
                        non_frozen_vars.append(var)
                        tf.logging.info('Training variable: %s', var.op.name)
                trainable_vars = non_frozen_vars
        else:
            tf.logging.info('Not initializing the model from a checkpoint. '
                            'Initializing from scratch!')

        # TODO(@oandrien): we might want to add gradient multiplier here
        # for the last layer if we have trouble with training
        # CPU of common ps server
        with tf.device(deploy_config.optimizer_device()):
            reg_losses = (None if train_config.add_regularization_loss
                               else [])
            if model_config.pspnet.train_reduce and reg_losses is None:
                regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                reg_losses = [r for r in regularization_losses if "dim_reduce" in r.name]
            
            total_loss, grads_and_vars = model_deploy.optimize_clones(
                clones, training_optimizer,
                regularization_losses=reg_losses,
                var_list=trainable_vars)
            # total_loss = tf.check_numerics(total_loss,
            #                               'total_loss is inf or nan.')
            summaries.add(
                tf.summary.scalar('Losses/TotalLoss', total_loss))
            # with tf.variable_scope("grad_clip"):
            #     grads_and_vars = [(tf.clip_by_norm(grad, 1.), var) for grad, var in grads_and_vars]
            grad_updates = training_optimizer.apply_gradients(grads_and_vars,
                                                    global_step=global_step)
            update_ops.append(grad_updates)
            update_op = tf.group(*update_ops, name='update_barrier')
            with tf.control_dependencies([update_op]):
                    train_op = tf.identity(total_loss, name='train_op')

        # TODO: this ideally should not be hardcoded like this.
        #   should have a way to access the prediction and GT tensor
        if image_summaries:
            graph = tf.get_default_graph()
            pixel_scaling = max(1, 255 // 19)
            summ_first_clone_scope = (first_clone_scope + '/'
                if first_clone_scope else '')
            input_img = graph.get_tensor_by_name(
                '%sInputs:0'% summ_first_clone_scope)
            main_labels = graph.get_tensor_by_name(
                '%sSegmentationLoss/ScaledLabels:0'% summ_first_clone_scope)
            main_preds = graph.get_tensor_by_name(
                '%sSegmentationLoss/ScaledPreds:0'% summ_first_clone_scope)
            summaries.add(
              tf.summary.image('VerifyTrainImages/Inputs', input_img))
            main_preds = tf.cast(tf.expand_dims(tf.argmax(main_preds, -1),-1) * pixel_scaling, tf.uint8)
            summaries.add(
              tf.summary.image('VerifyTrainImages/Predictions', main_preds))
            main_labels = tf.cast(main_labels * pixel_scaling, tf.uint8)
            summaries.add(
              tf.summary.image('VerifyTrainImages/Groundtruths', main_labels))

        # Add the summaries from the first clone. These contain the summaries
        # created by model_fn and either optimize_clones()
        # or _gather_clone_loss().
        summaries |= set(
            tf.get_collection(tf.GraphKeys.SUMMARIES, first_clone_scope))

        # Merge all summaries together.
        summary_op = tf.summary.merge(list(summaries))

        session_config = tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=False)
        #session_config.gpu_options.allow_growth = True

        #load_vars = [v for v in tf.global_variables() if "Dont_Load" not in v.op.name]

        # Save checkpoints regularly.
        saver = tf.train.Saver(max_to_keep=max_checkpoints_to_keep)

        # gpu_mems = []
        # for i in range(num_clones):
        #     with tf.device("/gpu:"+str(i)):
        #         gpu_mems.append(tf.cast(BytesInUse(), tf.float32)/float(1024*1024))

        # HACK to see memory usage.
        # TODO: Clean up, pretty messy.
        def train_step_mem(sess, train_op, global_step, train_step_kwargs):
            start_time = time.time()
            if log_memory:
                run_metadata = tf.RunMetadata()
                options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            else:
                run_metadata = None
                options = None
            total_loss, np_global_step, cur_gvs, dbg = sess.run([train_op, global_step, grads_and_vars, dist_builder.DEBUG],
                                        options=options,
                                        run_metadata=run_metadata)
            time_elapsed = time.time() - start_time
            #import pdb; pdb.set_trace()
            if 'should_log' in train_step_kwargs:
                if sess.run(train_step_kwargs['should_log']):
                    tf.logging.info(
                        'global step %d: loss = %.4f (%.3f sec/step)',
                        np_global_step, total_loss, time_elapsed)

            if log_memory:
                peaks = mem_util.peak_memory(run_metadata)
                for mem_use in peaks:
                # mem_use = mem_util.peak_memory(run_metadata)['/gpu:0']/1e6
                    if "/gpu" in mem_use:
                        tf.logging.info('Memory used (%s): %.2f MB', mem_use, peaks[mem_use]/1e6)
                # for m in mem:
                #     tf.logging.info('Memory used: %.2f MB',(m))

            if 'should_stop' in train_step_kwargs:
                should_stop = sess.run(train_step_kwargs['should_stop'])
            else:
                should_stop = False

            return total_loss, should_stop

        # Main training loop
        slim.learning.train(
            train_op,
            train_step_fn=train_step_mem,
            logdir=train_dir,
            master=master,
            is_chief=is_chief,
            session_config=session_config,
            number_of_steps=train_config.num_steps,
            startup_delay_steps=startup_delay_steps,
            init_fn=init_fn,
            summary_op=summary_op,
            save_summaries_secs=30,
            save_interval_secs=save_interval_secs,
            saver=saver)
