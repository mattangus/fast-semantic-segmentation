from experiment_mgr import db
import peewee as pw
from pydoc import locate

def create_config(run_args):
    configs = list(db.ExperimentConfig
           .select()
           .where((db.ExperimentConfig.model_config == run_args.model_config) &
                  (db.ExperimentConfig.data_config == run_args.data_config) &
                  (db.ExperimentConfig.trained_checkpoint == run_args.trained_checkpoint) &
                  (db.ExperimentConfig.pad_to_shape == run_args.pad_to_shape) &
                  (db.ExperimentConfig.processor_type == run_args.processor_type) &
                  (db.ExperimentConfig.annot_type == run_args.annot_type)))

    if len(configs) == 0:
        config = db.ExperimentConfig.create(
            model_config=run_args.model_config,
            data_config=run_args.data_config,
            trained_checkpoint=run_args.trained_checkpoint,
            pad_to_shape=run_args.pad_to_shape,
            processor_type=run_args.processor_type,
            annot_type=run_args.annot_type
        )
    elif len(configs) > 1:
        print("database has invalid data")
        import pdb; pdb.set_trace()
    else:
        config = configs[0]

    return config

def create_empty_args_group():
    arg_gs = list(db.ArgsGroup
        .select()
        .join(db.KeyWordArgs, pw.JOIN.LEFT_OUTER)
        .where(db.KeyWordArgs.id.is_null(True)))
    if len(arg_gs) == 0:
        arg_g = db.ArgsGroup.create()
    elif len(arg_gs) > 1:
        print("database has invalid data")
        import pdb; pdb.set_trace()
    else:
        arg_g = arg_gs[0]

    return arg_g

def create_args_group(run_args):
    if len(run_args.kwargs.items()) == 0:
        return create_empty_args_group()

    items = list(run_args.kwargs.items())
    query = (db.KeyWordArgs
            .select()
            .where((db.KeyWordArgs.name == items[0][0]) &
                   (db.KeyWordArgs.value == items[0][1])))

    #already queried for first
    for name, value in items[1:]:
        temp = db.KeyWordArgs.alias()
        query = (query.join(temp, on=(temp.group == db.KeyWordArgs.group))
                .where((temp.name == name) &
                   (temp.value == str(value)) &
                   (temp.value_type == type(value).__name__)
                )
            )

    subquery = (db.ArgsGroup
                .select(db.ArgsGroup, pw.fn.COUNT(db.ArgsGroup.id).alias('num'))
                .join(db.KeyWordArgs)
                .group_by(db.ArgsGroup.id))

    ids = [ag for ag in subquery if ag.num == len(items)]
    query = query.where(db.KeyWordArgs.group.in_(ids))

    kwargs = list(query)

    if len(kwargs) == 0:
        arg_g = db.ArgsGroup.create()
        for name, value in items:
            db.KeyWordArgs.create(
                group=arg_g,
                name=name,
                value=str(value),
                value_type=type(value).__name__
            )
    elif len(kwargs) > 1:
        print("database has invalid data")
        import pdb; pdb.set_trace()
    else:
        arg_g = (db.ArgsGroup
                 .select()
                 .where(db.ArgsGroup.id == kwargs[0].group)).get()

    return arg_g

def create_experiment(run_args):
    config = create_config(run_args)
    arg_g = create_args_group(run_args)

    experiments = list(
        db.Experiment
        .select()
        .where((db.Experiment.config == config) &
               (db.Experiment.arg_group == arg_g)
            )
        )
    
    if len(experiments) == 0:
        print("creating experiment")
        experiment = db.Experiment.create(
            config=config,
            arg_group=arg_g
        )
    elif len(experiments) > 1:
        print("database has invalid data")
        import pdb; pdb.set_trace()
    else:
        experiment = experiments[0]
    
    return experiment

def kwargs_to_dict(kwargs):
    ret = {}
    for kwarg in kwargs:
        if kwarg.value_type == "bool":
            val = kwarg.value == "True"
        else:
            val = locate(kwarg.value_type)(kwarg.value)
        ret[kwarg.name] = val
    
    return ret

def create_from_arg_list(arg_list):
    all_exps = map(create_experiment, arg_list)
    
    return all_exps

def get_top_from_config(config, exclude_fn=None):
    assert isinstance(config, db.ExperimentConfig), "config must be of type ExperimentConfig"

    results = list(
        db.Result.select()
        .join(db.Experiment)
        .where(db.Experiment.config == config)
    )

    if exclude_fn is not None:
        results = [r for r in results if not exclude_fn(r)]

    exps = set()
    for metric, ind in [("auroc",-1), ("aupr", -1), ("fpr_at_tpr",0), ("detection_error",0)]:
        ex = sorted(results, key=lambda x: x.__data__[metric])[ind]
        exps.add(ex)

    return list(exps)

def upload_result(run_args, print_buffer, result, had_error):
    values = result
    if values is None:
        values = {
            "auroc": 0,
            "aupr": 0,
            "fpr_at_tpr": 0,
            "detection_error": 0
        }
    
    experiment = create_experiment(run_args)
    exp_res = list(experiment.result)
    if len(exp_res) > 0 and not exp_res[0].had_error:
        print("uploading previous result")
        import pdb; pdb.set_trace()
    elif len(exp_res) == 1 and exp_res[0].had_error:
        print("updating result")
        result = exp_res[0]
        result.had_error = had_error
        result.auroc = values["auroc"]
        result.aupr = values["aupr"]
        result.fpr_at_tpr = values["fpr_at_tpr"]
        result.detection_error = values["detection_error"]
        result.save()
    elif len(exp_res) == 0:
        print("creating result")
        result = db.Result.create(
            experiment=experiment,
            had_error=had_error,
            **values
        )
        db.PrintBuffer.create(value=print_buffer, result=result)
    else:
        #can't happen
        raise Exception("this shouldn't happen")

#transfer from previous method
def _upload_legacy(results):
    for result in results:
        run_args = result[0]
        print_buffer, result, had_error = result[1]
        upload_result(run_args, print_buffer.getvalue(), result, had_error)