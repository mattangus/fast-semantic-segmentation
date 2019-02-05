from io import StringIO
import peewee as pw
import pickle

from experiment_mgr import db
from experiment_mgr import db_helper as dbh
from experiment_mgr import experiment_builder as eb

def get_all_to_run():
    builders = [eb.MaxSoftmaxRunBuilder(), eb.MahalRunBuilder()]

    to_run = []
    for builder in builders:
        to_run.extend(builder.to_run())
    
    return to_run