from io import StringIO
import peewee as pw
import pickle

from experiment_mgr import db
from experiment_mgr import db_helper as dbh
from experiment_mgr import experiment_builder as eb

def get_all_to_run():

    builder_classes = [eb.MaxSoftmaxRunBuilder,  eb.MahalRunBuilder, eb.ODINRunBuilder, eb.DropoutRunBuilder, eb.ConfidenceRunBuilder, eb.EntropyRunBuilder]
    # builder_classes = [eb.MaxSoftmaxRunBuilder,  eb.MahalRunBuilder, eb.DropoutRunBuilder, eb.ConfidenceRunBuilder, eb.EntropyRunBuilder]
    # builder_classes = [eb.MaxSoftmaxRunBuilder, eb.DropoutRunBuilder, eb.ConfidenceRunBuilder, eb.EntropyRunBuilder]
    #don't use mahal for small gpus
    #builder_classes = [eb.MaxSoftmaxRunBuilder, eb.MahalRunBuilder, eb.DropoutRunBuilder, eb.ConfidenceRunBuilder]
    #datasets = [("ood", eb.sun_experiment_set), ("error", eb.city_experiment_set), ("ood", eb.normal_experiment_set), ("ood", eb.uniform_experiment_set)]
    #only run ood experiments first
    datasets = [("ood", eb.sun_experiment_set), ("ood", eb.normal_experiment_set), ("ood", eb.uniform_experiment_set), ("ood", eb.perlin_experiment_set), ("ood", eb.coco_experiment_set)]

    builders = []
    for bc in builder_classes:
        for ds_type, ds in datasets:
            builders.append(bc(ds_type, ds))

    to_run = []
    for builder in builders:
        to_run.extend(builder.to_run())
    
    return to_run