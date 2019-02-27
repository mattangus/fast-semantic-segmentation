from flask import Flask
from flask import request
from queue import Queue

from experiment_mgr import experiment_factory
from experiment_mgr.experiment import Experiment, Result
from experiment_mgr import db
from experiment_mgr import db_helper as dbh

app = Flask("server")

common_queue = Queue()
start_size = 1

def _get():
    global common_queue #ew
    global start_size #ew
    #return next in queue
    if common_queue.empty():
        print("queue empty... filling")
        for to_run in experiment_factory.get_all_to_run():
            common_queue.put_nowait(to_run)
        print("done filling")
        start_size = common_queue.qsize()
        print("size:", start_size)
        if common_queue.empty():
            #still empty
            return "done"
    cur = common_queue.get_nowait()
    pct = (1 - (common_queue.qsize()/start_size))*100
    print("{0:.2f} percent done".format(pct))
    return cur.serialze()

def _post():
    if "results" not in request.form or "experiment" not in request.form:
        print("Something wrong with POST request")
        return "error"
    res = Result.deserialze(request.form["results"])
    exp = Experiment.deserialze(request.form["experiment"])
    
    dbh.upload_result(exp, res.buff, res.result, res.had_error)
    #print(exp, res.buff, res.result, res.had_error)

    return "success"

@app.route("/doresult", methods=["GET", "POST"])
def default():
    if request.method == "POST":
        return _post()
    elif request.method == "GET":
        return _get()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=False)