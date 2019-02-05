from . import experiment_runner
from multiprocessing import Process, Pool, Manager
from io import StringIO
from functools import partial
import pickle

from . import experiment_factory
from . import db_helper as dbh

# class Experiment(object):

#     def __init__(self, buffer=None):
#         self.model_config = None
#         self.data_config = None
#         self.trained_checkpoint = None
#         self.pad_to_shape = None
#         self.processor_type = None
#         self.annot_type = None
#         self.kwargs = None
#         if buffer is None:
#             self.print_buffer = StringIO()
#         else:
#             self.print_buffer = None

# #transfer from previous method
# def _upload_from_file(file):
#     with open(file, "rb") as f:
#         results = pickle.load(f)

#     dbh._upload_legacy(results)

# _upload_from_file("mahal_res.pkl")
# _upload_from_file("odin_res.pkl")
# _upload_from_file("topmahal_res.pkl")
# _upload_from_file("topodin_res.pkl")

def launch_experiment(exp, q):
    gpus = q.get()
    print("launching", exp.kwargs, "gpu", gpus)
    res = experiment_runner.run_experiment(gpus, exp.print_buffer, exp.model_config, exp.data_config,
                    exp.trained_checkpoint, exp.pad_to_shape,
                    exp.processor_type, exp.annot_type, **exp.kwargs)
    print("adding", gpus)
    q.put(gpus)
    return res

def main():
    pool = Pool(8)
    m = Manager()
    gpu_queue = m.Queue()
    for a in range(8):
        gpu_queue.put(str(a))

    to_run = experiment_factory.get_all_to_run()
    while len(to_run) > 0:
        exp_results = []
        
        for exp in to_run:
            res = pool.apply_async(launch_experiment, (exp, gpu_queue))
            exp_results.append((exp, res))

        for exp, res in exp_results:
            print_buffer, result, had_error = res.get()
            dbh.upload_result(exp, print_buffer.getvalue(), result, had_error)
        
        to_run = experiment_factory.get_all_to_run()
