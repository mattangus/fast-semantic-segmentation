from . import experiment_runner
from multiprocessing import Process, Pool, Manager
from io import StringIO
from functools import partial
import pickle

from . import experiment_factory
from . import db_helper as dbh

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
