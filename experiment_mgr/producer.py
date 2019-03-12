from . import experiment_runner
from multiprocessing import Process, Pool, Manager
from functools import partial
import pickle
import tqdm

from . import experiment_factory
from . import db_helper as dbh
from protos.experiment_pb2 import Experiment

#transfer from previous method
# def _upload_from_file(file):
#     with open(file, "rb") as f:
#         results = pickle.load(f)

#     dbh._upload_legacy(results)

def launch_experiment(exp, q, is_debug):
    gpus = q.get()
    print("launching", exp.kwargs, "gpu", gpus)
    res = experiment_runner.run_experiment(gpus, exp.model_config, exp.data_config,
                    exp.trained_checkpoint, exp.pad_to_shape,
                    exp.processor_type, exp.annot_type, is_debug, **exp.kwargs)
    print("adding", gpus)
    q.put(gpus)
    return res

def main(gpus, is_debug):
    # import pdb; pdb.set_trace()
    pool = Pool(len(gpus))
    m = Manager()
    gpu_queue = m.Queue()
    for a in gpus:
        gpu_queue.put(str(a))

    #TODO: make one gpu debugging better
    def one_gpu_launch(exp, gpu_queue, is_debug):
        return launch_experiment(exp,gpu_queue,is_debug)

    def one_gpu_get(res):
        return res

    def multi_gpu_launch(exp, gpu_queue, is_debug):
        return pool.apply_async(launch_experiment, (exp, gpu_queue,is_debug))

    def multi_gpu_get(res):
        return res.get()

    if len(gpus) == 1 or is_debug:
        launch = one_gpu_launch
        get = one_gpu_get
    else:
        launch = multi_gpu_launch
        get = multi_gpu_get

    to_run = experiment_factory.get_all_to_run()

    if is_debug:
        to_run = to_run[:1]

    while len(to_run) > 0:
        exp_results = []
        
        for exp in to_run:
            res = launch(exp, gpu_queue, is_debug)
            exp_results.append((exp, res))

        for exp, res in tqdm.tqdm(exp_results):
            print_buffer, result, had_error = get(res)
            buff = print_buffer.getvalue()
            dbh.upload_result(exp, buff, result, had_error)
            # if had_error:
            #     print(buff)
        
        to_run = experiment_factory.get_all_to_run()
