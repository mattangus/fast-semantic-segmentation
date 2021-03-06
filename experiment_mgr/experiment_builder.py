from io import StringIO
from abc import abstractmethod
import collections

from experiment_mgr.experiment import Experiment, ExperimentDataset
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

#ood
sun_experiment_set = ExperimentDataset("configs/data/sun_train.config", "configs/data/sun_eval.config")
idd_experiment_set = ExperimentDataset("configs/data/idd_cars_train.config", "configs/data/idd_cars_eval.config")
coco_experiment_set = ExperimentDataset("configs/data/coco_train.config", "configs/data/coco_eval.config")
coco_city_experiment_set = ExperimentDataset("configs/data/coco_city_train.config", "configs/data/coco_city_eval.config")
lost_experiment_set = ExperimentDataset("configs/data/lostfound_train.config", "configs/data/lostfound_eval.config")
normal_experiment_set = ExperimentDataset("configs/data/normal_train.config", "configs/data/normal_eval.config")
uniform_experiment_set = ExperimentDataset("configs/data/uniform_train.config", "configs/data/uniform_eval.config")
perlin_experiment_set = ExperimentDataset("configs/data/perlin_train.config", "configs/data/perlin_eval.config")
#error
city_experiment_set = ExperimentDataset("configs/data/cityscapes_train.config", "configs/data/cityscapes_eval.config")

RunModel = collections.namedtuple('modle_config', 'trained_checkpoint')

run_models = {
    "maxsoftmax": {
        "pspnet": ("configs/model/pspnet_full_dim.config", "remote/train_logs/resnet_dim/model.ckpt-1272"),
        "deeplab": ("configs/model/deeplab_full.config", "remote/train_logs/deeplab/model.ckpt-0")
    },
    "odin": {
        "pspnet": ("configs/model/pspnet_full_dim.config", "remote/train_logs/resnet_dim/model.ckpt-1272"),
        "deeplab": ("configs/model/deeplab_full.config", "remote/train_logs/deeplab/model.ckpt-0")
    },
    "mahal": {
        "pspnet": ("configs/model/pspnet_full_dim.config", "remote/train_logs/resnet_dim/model.ckpt-1272"),
        "deeplab": ("configs/model/deeplab_dim.config", "remote/train_logs/deeplab_dim/model.ckpt-4984")
    },
    "dropout": {
        "pspnet": ("configs/model/pspnet_dropout.config", "remote/train_logs/dropout/model.ckpt-31273"),
        "deeplab": ("configs/model/deeplab_dropout.config", "remote/train_logs/deeplab_dropout/model.ckpt-48780")
    },
    "conf": {
        "pspnet": ("configs/model/pspnet_confidence.config", "remote/train_logs/confidence/model.ckpt-13062"),
        "deeplab": ("configs/model/deeplab_confidence.config", "remote/train_logs/deeplab_conf/model.ckpt-8428")
    },
    "ent": {
        "pspnet": ("configs/model/pspnet_full_dim.config", "remote/train_logs/resnet_dim/model.ckpt-1272"),
        "deeplab": ("configs/model/deeplab_full.config", "remote/train_logs/deeplab/model.ckpt-0")
    },
    "alent": {
        "pspnet": ("configs/model/pspnet_dropout.config", "remote/train_logs/dropout/model.ckpt-31273"),
        "deeplab": ("configs/model/deeplab_dropout.config", "remote/train_logs/deeplab_dropout/model.ckpt-48780")
    },
}
    
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

    def __init__(self, annot_type, experiment_set, model_name):
        assert annot_type in ["ood", "error"]
        assert model_name in ["deeplab", "pspnet"]
        self.annot_type = annot_type
        self.experiment_set = experiment_set
        self.run_model = run_models["maxsoftmax"][model_name]

    @doc_inherit
    def make_args(self, epsilon, t_value, train=True):
        model_config = self.run_model[0]
        trained_checkpoint = self.run_model[1]
        data_config = self.experiment_set.eval_set
        pad_to_shape = "1025,2049"
        processor_type = "MaxSoftmax"
        annot_type = self.annot_type
        kwargs = {
            "epsilon": epsilon,
            "t_value": t_value,
        }
        run_args = Experiment(model_config, data_config,
                trained_checkpoint, pad_to_shape,
                processor_type, annot_type, kwargs)
        return run_args

    @doc_inherit
    def get_train(self):
        arg_list = []
        T = 1
        epsilon = 0.0
        run_args = self.make_args(epsilon, T)
        arg_list.append(run_args)
        return arg_list
    
    @doc_inherit
    def top_exclude_fn(self, result):
        return False

class ODINRunBuilder(RunnerBuilder):

    def __init__(self, annot_type, experiment_set, model_name):
        assert annot_type in ["ood", "error"]
        assert model_name in ["deeplab", "pspnet"]
        self.annot_type = annot_type
        self.experiment_set = experiment_set
        self.run_model = run_models["odin"][model_name]

    @doc_inherit
    def make_args(self, epsilon, t_value, train=True):
        model_config = self.run_model[0]
        trained_checkpoint = self.run_model[1]
        if train:
            data_config = self.experiment_set.train_set
        else:
            data_config = self.experiment_set.eval_set
        pad_to_shape = "1025,2049"
        processor_type = "ODIN"
        annot_type = self.annot_type
        kwargs = {
            "epsilon": epsilon,
            "t_value": t_value,
        }
        run_args = Experiment(model_config, data_config,
                trained_checkpoint, pad_to_shape,
                processor_type, annot_type, kwargs)
        return run_args

    @doc_inherit
    def get_train(self):
        arg_list = []
        epsilon = 0.0
        for T in [2, 5, 10, 20, 50, 100, 200, 500, 1000]:
            #for epsilon in [0.0, 0.00002, 0.00004, 0.00006, 0.00008, 0.00010, 0.00012, 0.00014, 0.00016, 0.00018, 0.00020, 0.00022, 0.00024, 0.00026, 0.00028, 0.00030, 0.00032, 0.00034, 0.00036, 0.00038, 0.00040]:
            if T == 1 and epsilon == 0.0:
                continue
            run_args = self.make_args(epsilon, T)
            arg_list.append(run_args)
        return arg_list
    
    @doc_inherit
    def top_exclude_fn(self, result):
        kwargs = dbh.kwargs_to_dict(list(result.experiment.arg_group.kwargs))
        eps = kwargs["epsilon"]
        t_value = kwargs["t_value"]
        return eps == 0.0 and t_value == 1

class MahalRunBuilder(RunnerBuilder):
    
    def __init__(self, annot_type, experiment_set, model_name):
        assert annot_type in ["ood", "error"]
        assert model_name in ["deeplab", "pspnet"]
        self.annot_type = annot_type
        self.experiment_set = experiment_set
        self.run_model = run_models["mahal"][model_name]
    
    @doc_inherit
    def make_args(self, epsilon,
            eval_dir="remote/eval_logs/resnet_dim/",
            global_cov=True, global_mean=False, train=True):
        model_config = self.run_model[0]
        trained_checkpoint = self.run_model[1]
        if train:
            data_config = self.experiment_set.train_set
        else:
            data_config = self.experiment_set.eval_set
        pad_to_shape = "1025,2049"
        processor_type = "Mahal"
        annot_type = self.annot_type
        kwargs = {
            "epsilon": epsilon,
            "eval_dir": eval_dir,
            "global_cov": global_cov,
            "global_mean": global_mean,
        }
        run_args = Experiment(model_config, data_config,
                trained_checkpoint, pad_to_shape,
                processor_type, annot_type, kwargs)
        return run_args

    @doc_inherit
    def get_train(self):
        arg_list = []
        epsilon = 0.0
        # for epsilon in [0.0, 0.00002, 0.00004, 0.00006, 0.00008, 0.00010, 0.00012, 0.00014, 0.00016, 0.00018, 0.00020, 0.00022, 0.00024, 0.00026, 0.00028, 0.00030, 0.00032, 0.00034, 0.00036, 0.00038, 0.00040]:
        run_args = self.make_args(epsilon)
        arg_list.append(run_args)
        return arg_list
    
    @doc_inherit
    def top_exclude_fn(self, result):
        return False

class DropoutRunBuilder(RunnerBuilder):

    def __init__(self, annot_type, experiment_set, model_name):
        assert annot_type in ["ood", "error"]
        assert model_name in ["deeplab", "pspnet"]
        self.annot_type = annot_type
        self.experiment_set = experiment_set
        self.run_model = run_models["dropout"][model_name]

    @doc_inherit
    def make_args(self, num_runs, train=True):
        model_config = self.run_model[0]
        trained_checkpoint = self.run_model[1]
        data_config = self.experiment_set.eval_set
        pad_to_shape = "1025,2049"
        processor_type = "Dropout"
        annot_type = self.annot_type
        kwargs = {
            "num_runs": num_runs,
        }
        run_args = Experiment(model_config, data_config,
                trained_checkpoint, pad_to_shape,
                processor_type, annot_type, kwargs)
        return run_args

    @doc_inherit
    def get_train(self):
        return [self.make_args(n) for n in [10]]
    
    @doc_inherit
    def top_exclude_fn(self, result):
        return False

class ConfidenceRunBuilder(RunnerBuilder):

    def __init__(self, annot_type, experiment_set, model_name):
        assert annot_type in ["ood", "error"]
        assert model_name in ["deeplab", "pspnet"]
        self.annot_type = annot_type
        self.experiment_set = experiment_set
        self.run_model = run_models["conf"][model_name]

    @doc_inherit
    def make_args(self, epsilon, train=True):
        model_config = self.run_model[0]
        trained_checkpoint = self.run_model[1]
        if train:
            data_config = self.experiment_set.train_set
        else:
            data_config = self.experiment_set.eval_set
        pad_to_shape = "1025,2049"
        processor_type = "Confidence"
        annot_type = self.annot_type
        kwargs = {
            "epsilon": epsilon,
        }
        run_args = Experiment(model_config, data_config,
                trained_checkpoint, pad_to_shape,
                processor_type, annot_type, kwargs)
        return run_args

    @doc_inherit
    def get_train(self):
        arg_list = []
        epsilon = 0.0
        # for epsilon in [0.0, 0.00002, 0.00004, 0.00006, 0.00008, 0.00010, 0.00012, 0.00014, 0.00016, 0.00018, 0.00020, 0.00022, 0.00024, 0.00026, 0.00028, 0.00030, 0.00032, 0.00034, 0.00036, 0.00038, 0.00040]:
        run_args = self.make_args(epsilon)
        arg_list.append(run_args)
        return arg_list
    
    @doc_inherit
    def top_exclude_fn(self, result):
        return False

class EntropyRunBuilder(RunnerBuilder):

    def __init__(self, annot_type, experiment_set, model_name):
        assert annot_type in ["ood", "error"]
        assert model_name in ["deeplab", "pspnet"]
        self.annot_type = annot_type
        self.experiment_set = experiment_set
        self.run_model = run_models["ent"][model_name]

    @doc_inherit
    def make_args(self, train=True):
        model_config = self.run_model[0]
        trained_checkpoint = self.run_model[1]
        data_config = self.experiment_set.eval_set
        pad_to_shape = "1025,2049"
        processor_type = "Entropy"
        annot_type = self.annot_type
        kwargs = {
        }
        run_args = Experiment(model_config, data_config,
                trained_checkpoint, pad_to_shape,
                processor_type, annot_type, kwargs)
        return run_args

    @doc_inherit
    def get_train(self):
        arg_list = []
        run_args = self.make_args()
        arg_list.append(run_args)
        return arg_list
    
    @doc_inherit
    def top_exclude_fn(self, result):
        return False

class AlEntRunBuilder(RunnerBuilder):

    def __init__(self, annot_type, experiment_set, model_name):
        assert annot_type in ["ood", "error"]
        assert model_name in ["deeplab", "pspnet"]
        self.annot_type = annot_type
        self.experiment_set = experiment_set
        self.run_model = run_models["alent"][model_name]

    # @doc_inherit
    def make_args(self, num_runs, train=True):
        model_config = self.run_model[0]
        trained_checkpoint = self.run_model[1]
        data_config = self.experiment_set.eval_set
        pad_to_shape = "1025,2049"
        processor_type = "AlEnt"
        annot_type = self.annot_type
        kwargs = {
            "num_runs": num_runs,
        }
        run_args = Experiment(model_config, data_config,
                trained_checkpoint, pad_to_shape,
                processor_type, annot_type, kwargs)
        
        return run_args

    @doc_inherit
    def get_train(self):
        return [self.make_args(n) for n in [8]]
    
    @doc_inherit
    def top_exclude_fn(self, result):
        return False