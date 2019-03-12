from experiment_mgr import image_extractor as ie

drop = 0
conf = 1
mahal = 2
softmax = 3
odin = 4

mode = conf

gpus = "2"
is_debug = True

if mode == drop:
    model_config = "configs/model/pspnet_dropout.config"
    data_config = "configs/data/sun_eval.config"
    trained_checkpoint = "remote/train_logs/dropout/model.ckpt-31273"
    pad_to_shape = "1025,2049"
    processor_type = "Dropout"
    annot_type = "ood"
    kwargs = {"num_runs": 8,}

    ie.extract_images(gpus, model_config, data_config,
                        trained_checkpoint, pad_to_shape,
                        processor_type, annot_type, is_debug, **kwargs)

elif mode == conf:
    model_config = "configs/model/pspnet_confidence.config"
    data_config = "configs/data/sun_eval.config"
    trained_checkpoint = "remote/train_logs/confidence/model.ckpt-13062"
    pad_to_shape = "1025,2049"
    processor_type = "Confidence"
    annot_type = "ood"
    kwargs = {"epsilon": 0.01}

    ie.extract_images(gpus, model_config, data_config,
                        trained_checkpoint, pad_to_shape,
                        processor_type, annot_type, is_debug, **kwargs)

elif mode == mahal:
    eval_dir = "remote/eval_logs/resnet_dim/"

    model_config = "configs/model/pspnet_full_dim.config"
    data_config = "configs/data/sun_eval.config"
    trained_checkpoint = "remote/train_logs/resnet_dim/model.ckpt-1272"
    pad_to_shape = "1025,2049"
    processor_type = "Mahal"
    annot_type = "ood"
    kwargs = {"epsilon": 0.0, "eval_dir": eval_dir, "global_cov": True, "global_mean": False,}

    ie.extract_images(gpus, model_config, data_config,
                        trained_checkpoint, pad_to_shape,
                        processor_type, annot_type, is_debug, **kwargs)

elif mode == softmax:
    model_config = "configs/model/pspnet_full_dim.config"
    data_config = "configs/data/uniform_train.config"
    trained_checkpoint = "remote/train_logs/resnet_dim/model.ckpt-1272"
    pad_to_shape = "1025,2049"
    processor_type = "MaxSoftmax"
    annot_type = "ood"
    kwargs = {"epsilon": 0.0, "t_value": 1}

    ie.extract_images(gpus, model_config, data_config,
                        trained_checkpoint, pad_to_shape,
                        processor_type, annot_type, is_debug, **kwargs)

elif mode == odin:
    model_config = "configs/model/pspnet_full_dim.config"
    data_config = "configs/data/uniform_train.config"
    trained_checkpoint = "remote/train_logs/resnet_dim/model.ckpt-1272"
    pad_to_shape = "1025,2049"
    processor_type = "MaxSoftmax"
    annot_type = "ood"
    kwargs = {"epsilon": 0.01, "t_value": 100}

    ie.extract_images(gpus, model_config, data_config,
                        trained_checkpoint, pad_to_shape,
                        processor_type, annot_type, is_debug, **kwargs)