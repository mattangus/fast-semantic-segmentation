from io import StringIO
import peewee as pw
import pickle

from experiment_mgr import db
from experiment_mgr import db_helper as dbh
from experiment_mgr import experiment_builder as eb

def get_all_to_run():
    builders = [
        # eb.MaxSoftmaxRunBuilder("ood", eb.normal_experiment_set),
        # eb.MahalRunBuilder("ood", eb.normal_experiment_set),
        # eb.DropoutRunBuilder("ood", eb.normal_experiment_set),
        # eb.ConfidenceRunBuilder("ood", eb.normal_experiment_set),


        # eb.MaxSoftmaxRunBuilder("ood", eb.uniform_experiment_set),
        # eb.MahalRunBuilder("ood", eb.uniform_experiment_set),
        # eb.DropoutRunBuilder("ood", eb.uniform_experiment_set),
        # eb.ConfidenceRunBuilder("ood", eb.uniform_experiment_set),


        eb.MaxSoftmaxRunBuilder("ood", eb.sun_experiment_set),
        eb.MahalRunBuilder("ood", eb.sun_experiment_set),
        eb.DropoutRunBuilder("ood", eb.sun_experiment_set),
        eb.ConfidenceRunBuilder("ood", eb.sun_experiment_set),


        eb.MaxSoftmaxRunBuilder("error", eb.city_experiment_set),
        eb.MahalRunBuilder("error", eb.city_experiment_set),
        eb.DropoutRunBuilder("error", eb.city_experiment_set),
        eb.ConfidenceRunBuilder("error", eb.city_experiment_set),
    ]

    # builders = [
    #     eb.DropoutRunBuilder("ood", eb.sun_experiment_set)
    # ]

    to_run = []
    for builder in builders:
        to_run.extend(builder.to_run())
    
    return to_run