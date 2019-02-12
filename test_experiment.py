from experiment_mgr import experiment_runner as er

model_config = "configs/model/pspnet_dropout.config"
data_config = "configs/data/sun_eval.config"
trained_checkpoint = "remote/train_logs/dropout/model.ckpt-31273"
pad_to_shape = "1025,2049"
processor_type = "Dropout"
annot_type = "ood"
kwargs = {"num_runs": 6,}
is_debug = True

er.run_experiment("2,3", None, model_config, data_config,
                    trained_checkpoint, pad_to_shape,
                    processor_type, annot_type, is_debug, **kwargs)

# model_config = "configs/model/pspnet_confidence.config"
# data_config = "configs/data/sun_eval.config"
# trained_checkpoint = "remote/train_logs/confidence/model.ckpt-13062"
# pad_to_shape = "1025,2049"
# processor_type = "Confidence"
# annot_type = "ood"
# kwargs = {"epsilon": 0.01}
# is_debug = True

# er.run_experiment("2", None, model_config, data_config,
#                     trained_checkpoint, pad_to_shape,
#                     processor_type, annot_type, is_debug, **kwargs)