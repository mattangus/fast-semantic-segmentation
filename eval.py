r"""Main Evaluation script for ICNet"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import functools
import tensorflow as tf

from google.protobuf import text_format

from builders import dataset_builder
from builders import model_builder
from protos import pipeline_pb2
from libs.evaluator import eval_segmentation_model
from libs.evaluator import eval_segmentation_model_once
from protos.config_reader import read_config

tf.logging.set_verbosity(tf.logging.INFO)


flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_string('evaluate_all_from_checkpoint', None,
                    'FIlename to a checkpoint from which to begin running '
                    'evaluation from. All proceeding checkpoints will also '
                    ' be evaluated. Should be the similar to model.ckpt.XXX')

flags.DEFINE_string('train_dir', None,
                    'Directory containing checkpoints to evaluate, typically '
                    'set to `train_dir` used in the training job.')
flags.mark_flag_as_required('train_dir')

flags.DEFINE_string('eval_dir', None,
                    'Directory to write eval summaries to.')
flags.mark_flag_as_required('eval_dir')

flags.DEFINE_string('model_config', None,
                    'Path to a pipeline_pb2.TrainEvalConfig config '
                    'file. If provided, other configs are ignored')
flags.mark_flag_as_required('model_config')

flags.DEFINE_string('data_config', None,
                    'Path to a pipeline_pb2.TrainEvalConfig config '
                    'file. If provided, other configs are ignored')
flags.mark_flag_as_required('data_config')

flags.DEFINE_boolean('image_summaries', False,
                     'Show summaries of eval predictions and save to '
                     'Tensorboard for viewing.')

flags.DEFINE_boolean('verbose', False,
                     'Show streamed mIoU updates during evaluation of '
                     'various checkpoint evaluation runs.')


def get_checkpoints_from_path(initial_checkpoint_path, checkpoint_dir):
    checkpoints = tf.train.get_checkpoint_state(checkpoint_dir)
    if checkpoints is None:
        raise ValueError('No checkpoints found in `train_dir`.')
    all_checkpoints = checkpoints.all_model_checkpoint_paths
    # Loop through all checkpoints in the specified dir
    checkpoints_to_evaluate = None
    tf.logging.info('Searching checkpoints in %s', checkpoint_dir)
    for idx, ckpt in enumerate(all_checkpoints):
        print("skipping:", idx, ' ', str(ckpt))
        dirname = os.path.dirname(ckpt)
        full_init_ckpt_path = os.path.join(dirname, initial_checkpoint_path)
        if str(ckpt) == full_init_ckpt_path:
            checkpoints_to_evaluate = all_checkpoints[idx:]
            break
    # We must be able to find the checkpoint specified
    if checkpoints_to_evaluate is None:
        raise ValueError('Checkpoint not found. Exiting.')
    return checkpoints_to_evaluate


def main(_):
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    if not tf.gfile.IsDirectory(FLAGS.train_dir):
        raise ValueError('`train_dir` must be a valid directory '
                         'containing model checkpoints from training.')
    pipeline_config = read_config(FLAGS.model_config, FLAGS.data_config)
    eval_config = pipeline_config.eval_config
    input_config = pipeline_config.input_reader
    model_config = pipeline_config.model

    #TODO:handle this special case better
    class_loss_type = model_config.pspnet.loss.classification_loss.WhichOneof('loss_type')
    if class_loss_type == "confidence":
        num_extra_class = 1
    else:
        num_extra_class = 0
        
    create_input_fn = functools.partial(
        dataset_builder.build,
        input_reader_config=input_config)
    create_model_fn = functools.partial(
        model_builder.build,
        model_config=model_config,
        is_training=False,
        ignore_label=input_config.ignore_label)

    eval_input_type = eval_config.eval_input_type
    input_type = eval_input_type.WhichOneof('eval_input_type_oneof')
    if input_type == 'cropped_eval_input':
        cropped_eval_input = eval_input_type.cropped_eval_input
        input_dims = (cropped_eval_input.height,
                      cropped_eval_input.width)
        cropped_evaluation = True
    elif input_type == 'padded_eval_input':
        padded_eval_input = eval_input_type.padded_eval_input
        input_dims = (padded_eval_input.height,
                      padded_eval_input.width)
        cropped_evaluation = False
    else:
        raise ValueError('Must specify an `eval_input_type` for evaluation.')

    if FLAGS.evaluate_all_from_checkpoint is not None:
        checkpoints_to_evaluate = get_checkpoints_from_path(
            FLAGS.evaluate_all_from_checkpoint, FLAGS.train_dir)
        # Run eval on each checkpoint only once. Exit when done.
        for curr_checkpoint in checkpoints_to_evaluate:
            tf.reset_default_graph()
            eval_segmentation_model_once(curr_checkpoint,
                                         create_model_fn,
                                         create_input_fn,
                                         input_dims,
                                         eval_config,
                                         num_extra_class=num_extra_class,
                                         input_reader=input_config,
                                         eval_dir=FLAGS.eval_dir,
                                         cropped_evaluation=cropped_evaluation,
                                         image_summaries=FLAGS.image_summaries,
                                         verbose=FLAGS.verbose)
    else:
        eval_segmentation_model(
            create_model_fn,
            create_input_fn,
            input_dims,
            eval_config,
            num_extra_class=num_extra_class,
            input_reader=input_config,
            train_dir=FLAGS.train_dir,
            eval_dir=FLAGS.eval_dir,
            cropped_evaluation=cropped_evaluation,
            evaluate_single_checkpoint=FLAGS.evaluate_all_from_checkpoint,
            image_summaries=FLAGS.image_summaries,
            verbose=FLAGS.verbose)

if __name__ == '__main__':
    tf.app.run()
