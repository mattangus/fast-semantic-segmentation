import requests
import argparse
from experiment_mgr import experiment_runner
import multiprocessing as mp
from functools import partial
import pickle
import tqdm
from enum import Enum
import time
import tensorflow as tf

from experiment_mgr.experiment import Experiment, Result

flags = tf.app.flags

flags.DEFINE_integer('port', 5000,'')
flags.DEFINE_string('host', None,'')

flags.DEFINE_list('gpus', ['0'],'')

flags.DEFINE_boolean('debug', False,'')

# parser = argparse.ArgumentParser()
# parser.add_argument("--host", type=str, required=True)
# parser.add_argument("--port", type=int, default=5000)
# parser.add_argument("gpus", nargs='+', type=str)
# parser.add_argument("--debug", action='store_true')

# args = parser.parse_args()

args = flags.FLAGS

try:
    for gpu in args.gpus:
        #allow cpu only
        if gpu != "":
            list(map(int,gpu.split(",")))
except:
    print("invalid gpu list")
    import sys; sys.exit(1)

url = "http://" + args.host + ":" + str(args.port) + "/doresult"

print("using:", url)

class Response(Enum):
    error = 1
    done = 2
    ok = 3

def _get_next_exp():
    r = requests.get(url)
    if r.ok:
        if r.text == "error":
            #sent bad data
            print("bad data sent")
            import pdb; pdb.set_trace()
            return Response.error, None
        elif r.text == "done":
            #we are finished.
            return Response.done, None
        return Response.ok, Experiment.deserialze(r.text)
    else:
        print("got error")
        print(r.text)
        import pdb; pdb.set_trace()
        return Response.error, None

def _upload_result(exp, buff, result, had_error):
    res_obj = Result(buff, result, had_error)
    res_str = res_obj.serialze()
    exp_str = exp.serialze()

    r = requests.post(url, data = {"experiment": exp_str, "results": res_str})

    if r.ok:
        if r.text == "success":
            return Response.ok
    
    print("error response")
    import pdb; pdb.set_trace()

def runner_thread(gpu_queue, exp_queue, res_queue, is_debug):
    
    while True:
        exp = exp_queue.get()
        if exp is None:
            return
        gpus = gpu_queue.get()
        print("launching", exp.model_config, "-", exp.data_config, "-", exp.kwargs, "-", exp.processor_type, "-", "gpu", gpus)
        buff, result, had_error = experiment_runner.run_experiment(gpus, exp.model_config, exp.data_config,
                        exp.trained_checkpoint, exp.pad_to_shape,
                        exp.processor_type, exp.annot_type, is_debug, **exp.kwargs)
        print("adding", gpus)
        gpu_queue.put(gpus)
        res_queue.put((exp, buff, result, had_error))

def _has_item(q):
    #will mess up the order of the queue
    #here order doesn't matter so this is fine
    val = None
    try:
        val = q.get(block=False)
        q.put(val)
    except:
        pass
    return val != None

def main(gpus, is_debug):
    # import pdb; pdb.set_trace()
    m = mp.Manager()
    gpu_queue = m.Queue()
    exp_queue = m.Queue()
    res_queue = m.Queue()

    if not is_debug:
        procs = []
        for _ in range(len(gpus)):
            proc = mp.Process(target=runner_thread, args=(gpu_queue, exp_queue, res_queue, is_debug))
            proc.start()
            procs.append(proc)

    for a in gpus:
        gpu_queue.put(str(a))
    
    is_done = False

    while True:
        if not is_done and _has_item(gpu_queue):
            #gpu is available so get a new experiment to run
            resp, exp = _get_next_exp()
            if resp == Response.ok:
                exp_queue.put(exp)
                if is_debug:
                    #call manually if debugging, subprocs not created
                    #this will never return.
                    runner_thread(gpu_queue, exp_queue, res_queue, is_debug)
            elif resp == Response.done:
                is_done = True
        
        if _has_item(res_queue):
            exp, buff, result, had_error = res_queue.get()
            buff = buff.getvalue()
            _upload_result(exp, buff, result, had_error)
        
        #chill for a bit
        time.sleep(5)

        
    if not is_debug:
        for proc in procs:
            exp_queue.put(None)
        
        for proc in procs:
            proc.join()

    #all processes have joined. get all results
    while _has_item(res_queue):
        exp, buff, result, had_error = res_queue.get()
        buff = buff.getvalue()
        _upload_result(exp, buff, result, had_error)

if __name__ == "__main__":
    main(args.gpus, args.debug)