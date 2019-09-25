import tensorflow as tf

from experiment_mgr import experiment_builder as eb
from experiment_mgr import image_extractor as ie
from experiment_mgr import db_helper as dbh

flags = tf.app.flags

flags.DEFINE_string('ptype', None,'')

flags.DEFINE_string("export_folder", None, "")

flags.DEFINE_string("arch", None, "")

flags.DEFINE_string("data_config", None, "")

flags.DEFINE_string('gpus', '0','')

flags.DEFINE_boolean('no_gpu', False,'')

FLAGS = flags.FLAGS

assert FLAGS.ptype in eb.run_models, str(FLAGS.ptype) + " must be one of " + str(list(eb.models.keys()))
run_model = eb.run_models[FLAGS.ptype]
assert FLAGS.arch in run_model, str(FLAGS.arch) + " must be one of " + str(list(run_model.keys()))
model_config, trained_checkpoint = run_model[FLAGS.arch]
data_config = FLAGS.data_config

for g in FLAGS.gpus.split(","):
    try:
        int(g)
    except ex:
        print("'" + g + "' is an invalid gpu number")
        exit(0)

drop = "dropout"
conf = "conf"
mahal = "mahal"
softmax = "maxsoftmax"
odin = "odin"
entropy = "ent"
alent = "alent"

mode = FLAGS.ptype
export_folder = FLAGS.export_folder

gpus = FLAGS.gpus
if FLAGS.no_gpu:
    gpus = ""
is_debug = True

def get_top(model_config, data_config, trained_checkpoint, pad_to_shape, processor_type, annot_type):
    config = dbh.get_config(model_config, data_config, trained_checkpoint, pad_to_shape, processor_type, annot_type)
    assert len(config) == 1, "wrong number of configs found: " + str(len(config))
    config = config[0] 
    top = dbh.get_top_from_config(config)
    assert len(top) > 0, "no top found"
    
    return dbh.kwargs_to_dict(list(top[0].experiment.arg_group.kwargs))


if mode == drop:
    pad_to_shape = "1025,2049"
    processor_type = "Dropout"
    annot_type = "ood"
    kwargs = {"num_runs": 8,}

    # ie.extract_images(gpus, model_config, data_config,
    #                     trained_checkpoint, pad_to_shape,
    #                     processor_type, annot_type, is_debug, **kwargs)

elif mode == conf:
    pad_to_shape = "1025,2049"
    processor_type = "Confidence"
    annot_type = "ood"
    kwargs = {"epsilon": 0.0}

    # ie.extract_images(gpus, model_config, data_config,
    #                     trained_checkpoint, pad_to_shape,
    #                     processor_type, annot_type, is_debug, **kwargs)

elif mode == mahal:
    eval_dir = "remote/eval_logs/resnet_dim/"
    pad_to_shape = "1025,2049"
    processor_type = "Mahal"
    annot_type = "ood"
    kwargs = {"epsilon": 0.0, "eval_dir": eval_dir, "global_cov": True, "global_mean": False,}

    # ie.extract_images(gpus, model_config, data_config,
    #                     trained_checkpoint, pad_to_shape,
    #                     processor_type, annot_type, is_debug, **kwargs)

elif mode == softmax:
    pad_to_shape = "1025,2049"
    processor_type = "MaxSoftmax"
    annot_type = "ood"
    kwargs = {"epsilon": 0.0, "t_value": 1}

    # ie.extract_images(gpus, model_config, data_config,
    #                     trained_checkpoint, pad_to_shape,
    #                     processor_type, annot_type, is_debug, **kwargs)

elif mode == odin:
    pad_to_shape = "1025,2049"
    processor_type = "ODIN"
    annot_type = "ood"
    kwargs = get_top(model_config, data_config, trained_checkpoint, pad_to_shape, processor_type, annot_type)
    #kwargs = {"epsilon": 0.0, "t_value": 10}
    print("using top kwargs:", kwargs)

    # ie.extract_images(gpus, model_config, data_config,
    #                     trained_checkpoint, pad_to_shape,
    #                     processor_type, annot_type, is_debug, **kwargs)

elif mode == entropy:
    pad_to_shape = "1025,2049"
    processor_type = "Entropy"
    annot_type = "ood"
    kwargs = {}

    # ie.extract_images(gpus, model_config, data_config,
    #                     trained_checkpoint, pad_to_shape,
    #                     processor_type, annot_type, is_debug, **kwargs)

elif mode == alent:
    pad_to_shape = "1025,2049"
    processor_type = "AlEnt"
    annot_type = "ood"
    kwargs = {"num_runs": 8,}

ie.extract_images(gpus, model_config, data_config,
                    trained_checkpoint, pad_to_shape,
                    processor_type, annot_type, is_debug,
                    export_folder, **kwargs)