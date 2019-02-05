from io import StringIO
import peewee as pw
import pickle

from experiment_mgr import db
from experiment_mgr import db_helper as dbh

class RunnerArgs(object):

    def __init__(self, buffer=None):
        self.model_config = None
        self.data_config = None
        self.trained_checkpoint = None
        self.pad_to_shape = None
        self.processor_type = None
        self.annot_type = None
        self.kwargs = None
        if buffer is None:
            self.print_buffer = StringIO()
        else:
            self.print_buffer = None

def _make_softmax_args(epsilon, t_value, train=True):
    run_args = RunnerArgs()
    run_args.model_config = "configs/model/pspnet_full_dim.config"
    if train:
        run_args.data_config = "configs/data/sun_train.config"
    else:
        run_args.data_config = "configs/data/sun_eval.config"
    run_args.trained_checkpoint = "remote/train_logs/resnet_dim/model.ckpt-1272"
    run_args.pad_to_shape = "1025,2049"
    run_args.processor_type = "MaxSoftmax"
    run_args.annot_type = "ood"
    run_args.kwargs = {
        "epsilon": epsilon,
        "t_value": t_value,
    }
    return run_args

def _make_mahal_args(epsilon,
        eval_dir="remote/eval_logs/resnet_dim/",
        global_cov=True, global_mean=False, train=True):
    run_args = RunnerArgs()
    run_args.model_config = "configs/model/pspnet_full_dim.config"
    if train:
        run_args.data_config = "configs/data/sun_train.config"
    else:
        run_args.data_config = "configs/data/sun_eval.config"
    run_args.trained_checkpoint = "remote/train_logs/resnet_dim/model.ckpt-1272"
    run_args.pad_to_shape = "1025,2049"
    run_args.processor_type = "Mahal"
    run_args.annot_type = "ood"
    run_args.kwargs = {
        "epsilon": epsilon,
        "eval_dir": eval_dir,
        "global_cov": global_cov,
        "global_mean": global_mean,
    }
    return run_args

def _max_softmax_train():
    arg_list = []
    for T in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]:
        for epsilon in [0.0, 0.00002, 0.00004, 0.00006, 0.00008, 0.00010, 0.00012, 0.00014, 0.00016, 0.00018, 0.00020, 0.00022, 0.00024, 0.00026, 0.00028, 0.00030, 0.00032, 0.00034, 0.00036, 0.00038, 0.00040]:
            run_args = _make_softmax_args(epsilon, T)
            arg_list.append(run_args)
    return arg_list

def _mahal_train():
    arg_list = []
    for epsilon in [0.0, 0.00002, 0.00004, 0.00006, 0.00008, 0.00010, 0.00012, 0.00014, 0.00016, 0.00018, 0.00020, 0.00022, 0.00024, 0.00026, 0.00028, 0.00030, 0.00032, 0.00034, 0.00036, 0.00038, 0.00040]:
        run_args = _make_mahal_args(epsilon)
        arg_list.append(run_args)
    return arg_list

def _filter_already_run(arg_list):
    all_exps = dbh.create_from_arg_list(arg_list)
    ret = []
    for experiment, run_args in zip(all_exps, arg_list):
        result = list(experiment.result)
        if len(result) > 1:
            print("database has invalid data")
            import pdb; pdb.set_trace()
        elif len(result) == 0:
            ret.append(run_args)
    
    return ret

#transfer from previous method
def _upload_legacy(results):
    for result in results:
        experiment = dbh.create_experiment(result[0])
        print_buffer, result, had_error = result[1]
        if len(experiment.result) > 0:
            print("uploading previous result")
            import pdb; pdb.set_trace()
        else:
            values = result
            if values is None:
                values = {
                    "auroc": 0,
                    "aupr": 0,
                    "fpr_at_tpr": 0,
                    "detection_error": 0
                }
            print("creating result")
            db.Result.create(
                experiment=experiment,
                print_buffer=print_buffer.getvalue(),
                had_error=had_error,
                **values
            )

def _all_to_run_from_builder(train_fn, make_fn, top_exclude_fn=None):
    arg_list = train_fn()
    filtered_list = _filter_already_run(arg_list)

    if len(filtered_list) == 0:
        #same config
        config = dbh.create_config(arg_list[0])
        arg_list = []
        top = dbh.get_top_from_config(config, exclude_fn=top_exclude_fn)
        for result in top:
            kwargs = dbh.kwargs_to_dict(list(result.experiment.arg_group.kwargs))
            arg_list.append(make_fn(train=False, **kwargs))
        filtered_list = _filter_already_run(arg_list)
    
    return filtered_list

def get_all_to_run():
    mahal_list = _all_to_run_from_builder(_mahal_train, _make_mahal_args)

    def exclude_baseline(result):
        eps = dbh.kwargs_to_dict(list(result.experiment.arg_group.kwargs))["epsilon"]
        return eps == 0.0
    max_softmax_list = _all_to_run_from_builder(_max_softmax_train, _make_softmax_args, exclude_baseline)

    to_run = []
    for v in [mahal_list, max_softmax_list]:
        to_run.extend(v)
    
    return to_run