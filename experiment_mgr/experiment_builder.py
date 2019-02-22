from io import StringIO
from abc import abstractmethod

from third_party.doc_inherit import doc_inherit
from . import db_helper as dbh

def _filter_already_run(arg_list):
    all_exps = dbh.create_from_arg_list(arg_list)
    ret = []
    for experiment, run_args in zip(all_exps, arg_list):
        result = list(experiment.result)
        if len(result) > 1:
            print("database has invalid data")
            import pdb; pdb.set_trace()
        elif len(result) == 0 or result[0].had_error:
            ret.append(run_args)
    
    return ret

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

class ExperimentDataset(object):

    def __init__(self, train_set, eval_set=None):
        self.train_set = train_set
        self.eval_set = eval_set
        if eval_set is None:
            self.eval_set = train_set

#ood
sun_experiment_set = ExperimentDataset("configs/data/sun_train.config", "configs/data/sun_eval.config")
normal_experiment_set = ExperimentDataset("configs/data/normal_train.config", "configs/data/normal_eval.config")
uniform_experiment_set = ExperimentDataset("configs/data/uniform_train.config", "configs/data/uniform_eval.config")
#error
city_experiment_set = ExperimentDataset("configs/data/cityscapes_train.config", "configs/data/cityscapes_eval.config")

class RunnerBuilder(object):
    
    @abstractmethod
    def make_args(self, train=True):
        """Create a new RunnerArgs object and populate
        
        Keyword Arguments:
            train {bool} -- use train dataset (default: {True})
        """
        pass

    @abstractmethod
    def get_train(self):
        """get a default train set of RunnerArgs
        """
        pass
    
    @abstractmethod
    def top_exclude_fn(self, result):
        """Get a function that accepts a result and returns if \
           it should be excluded from determining the top
        """
        pass
    
    def to_run(self):
        arg_list = self.get_train()
        filtered_list = _filter_already_run(arg_list)

        if len(filtered_list) == 0:
            #same config
            config = dbh.create_config(arg_list[0])
            arg_list = []
            top = dbh.get_top_from_config(config, exclude_fn=self.top_exclude_fn)
            for result in top:
                kwargs = dbh.kwargs_to_dict(list(result.experiment.arg_group.kwargs))
                arg_list.append(self.make_args(train=False, **kwargs))
            filtered_list = _filter_already_run(arg_list)
        
        return filtered_list


class MaxSoftmaxRunBuilder(RunnerBuilder):

    def __init__(self, annot_type, experiment_set):
        assert annot_type in ["ood", "error"]
        self.annot_type = annot_type
        self.experiment_set = experiment_set

    @doc_inherit
    def make_args(self, epsilon, t_value, train=True):
        run_args = RunnerArgs()
        run_args.model_config = "configs/model/pspnet_full_dim.config"
        if train:
            run_args.data_config = self.experiment_set.train_set
        else:
            run_args.data_config = self.experiment_set.eval_set
        run_args.trained_checkpoint = "remote/train_logs/resnet_dim/model.ckpt-1272"
        run_args.pad_to_shape = "1025,2049"
        run_args.processor_type = "MaxSoftmax"
        run_args.annot_type = self.annot_type
        run_args.kwargs = {
            "epsilon": epsilon,
            "t_value": t_value,
        }
        return run_args

    @doc_inherit
    def get_train(self):
        arg_list = []
        for T in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]:
            for epsilon in [0.0, 0.00002, 0.00004, 0.00006, 0.00008, 0.00010, 0.00012, 0.00014, 0.00016, 0.00018, 0.00020, 0.00022, 0.00024, 0.00026, 0.00028, 0.00030, 0.00032, 0.00034, 0.00036, 0.00038, 0.00040]:
                run_args = self.make_args(epsilon, T)
                arg_list.append(run_args)
        return arg_list
    
    @doc_inherit
    def top_exclude_fn(self, result):
        eps = dbh.kwargs_to_dict(list(result.experiment.arg_group.kwargs))["epsilon"]
        return eps == 0.0

class MahalRunBuilder(RunnerBuilder):
    
    def __init__(self, annot_type, experiment_set):
        assert annot_type in ["ood", "error"]
        self.annot_type = annot_type
        self.experiment_set = experiment_set
    
    @doc_inherit
    def make_args(self, epsilon,
            eval_dir="remote/eval_logs/resnet_dim/",
            global_cov=True, global_mean=False, train=True):
        run_args = RunnerArgs()
        run_args.model_config = "configs/model/pspnet_full_dim.config"
        if train:
            run_args.data_config = self.experiment_set.train_set
        else:
            run_args.data_config = self.experiment_set.eval_set
        run_args.trained_checkpoint = "remote/train_logs/resnet_dim/model.ckpt-1272"
        run_args.pad_to_shape = "1025,2049"
        run_args.processor_type = "Mahal"
        run_args.annot_type = self.annot_type
        run_args.kwargs = {
            "epsilon": epsilon,
            "eval_dir": eval_dir,
            "global_cov": global_cov,
            "global_mean": global_mean,
        }
        return run_args

    @doc_inherit
    def get_train(self):
        arg_list = []
        for epsilon in [0.0, 0.00002, 0.00004, 0.00006, 0.00008, 0.00010, 0.00012, 0.00014, 0.00016, 0.00018, 0.00020, 0.00022, 0.00024, 0.00026, 0.00028, 0.00030, 0.00032, 0.00034, 0.00036, 0.00038, 0.00040]:
            run_args = self.make_args(epsilon)
            arg_list.append(run_args)
        return arg_list
    
    @doc_inherit
    def top_exclude_fn(self, result):
        return False

class DropoutRunBuilder(RunnerBuilder):

    def __init__(self, annot_type, experiment_set):
        assert annot_type in ["ood", "error"]
        self.annot_type = annot_type
        self.experiment_set = experiment_set

    @doc_inherit
    def make_args(self, num_runs, train=True):
        run_args = RunnerArgs()
        run_args.model_config = "configs/model/pspnet_dropout.config"
        if train:
            run_args.data_config = self.experiment_set.train_set
        else:
            run_args.data_config = self.experiment_set.eval_set
        run_args.trained_checkpoint = "remote/train_logs/dropout/model.ckpt-31273"
        run_args.pad_to_shape = "1025,2049"
        run_args.processor_type = "Dropout"
        run_args.annot_type = self.annot_type
        run_args.kwargs = {
            "num_runs": num_runs,
        }
        return run_args

    @doc_inherit
    def get_train(self):
        return [self.make_args(n) for n in [4, 6, 8]]
    
    @doc_inherit
    def top_exclude_fn(self, result):
        return False

class ConfidenceRunBuilder(RunnerBuilder):

    def __init__(self, annot_type, experiment_set):
        assert annot_type in ["ood", "error"]
        self.annot_type = annot_type
        self.experiment_set = experiment_set

    @doc_inherit
    def make_args(self, epsilon, train=True):
        run_args = RunnerArgs()
        run_args.model_config = "configs/model/pspnet_confidence.config"
        if train:
            run_args.data_config = self.experiment_set.train_set
        else:
            run_args.data_config = self.experiment_set.eval_set
        run_args.trained_checkpoint = "remote/train_logs/confidence/model.ckpt-13062"
        run_args.pad_to_shape = "1025,2049"
        run_args.processor_type = "Confidence"
        run_args.annot_type = self.annot_type
        run_args.kwargs = {
            "epsilon": epsilon,
        }
        return run_args

    @doc_inherit
    def get_train(self):
        arg_list = []
        for epsilon in [0.0, 0.00002, 0.00004, 0.00006, 0.00008, 0.00010, 0.00012, 0.00014, 0.00016, 0.00018, 0.00020, 0.00022, 0.00024, 0.00026, 0.00028, 0.00030, 0.00032, 0.00034, 0.00036, 0.00038, 0.00040]:
            run_args = self.make_args(epsilon)
            arg_list.append(run_args)
        return arg_list
    
    @doc_inherit
    def top_exclude_fn(self, result):
        return False

# class MaxSoftmaxRunBuilder(RunnerBuilder):

#     @doc_inherit
#     def make_args(self, epsilon, t_value, train=True):
#         run_args = RunnerArgs()
#         run_args.model_config = "configs/model/pspnet_full_dim.config"
#         if train:
#             run_args.data_config = "configs/data/sun_train.config"
#         else:
#             run_args.data_config = "configs/data/sun_eval.config"
#         run_args.trained_checkpoint = "remote/train_logs/resnet_dim/model.ckpt-1272"
#         run_args.pad_to_shape = "1025,2049"
#         run_args.processor_type = "MaxSoftmax"
#         run_args.annot_type = "ood"
#         run_args.kwargs = {
#             "epsilon": epsilon,
#             "t_value": t_value,
#         }
#         return run_args

#     @doc_inherit
#     def get_train(self):
#         arg_list = []
#         for T in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]:
#             for epsilon in [0.0, 0.00002, 0.00004, 0.00006, 0.00008, 0.00010, 0.00012, 0.00014, 0.00016, 0.00018, 0.00020, 0.00022, 0.00024, 0.00026, 0.00028, 0.00030, 0.00032, 0.00034, 0.00036, 0.00038, 0.00040]:
#                 run_args = self.make_args(epsilon, T)
#                 arg_list.append(run_args)
#         return arg_list
    
#     @doc_inherit
#     def top_exclude_fn(self, result):
#         eps = dbh.kwargs_to_dict(list(result.experiment.arg_group.kwargs))["epsilon"]
#         return eps == 0.0