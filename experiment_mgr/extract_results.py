from pprint import pprint

from . import db
from . import db_helper as dbh

model_config = 0
data_config = 1
trained_checkpoint = 2
processor_type = 3
annot_type = 4
arg_group_id = 5
auroc = 6
aupr = 7
fpr_at_tpr = 8
detection_error = 9
max_iou = 10
had_error = 11

class ResultWrapper(object):

    def __init__(self, row):
        self.model_config =         row[0]
        self.data_config =          row[1]
        self.trained_checkpoint =   row[2]
        self.processor_type =       row[3]
        self.annot_type =           row[4]
        self.experiment_id =        row[5]
        self.arg_group_id =         row[6]
        self.auroc =                row[7]
        self.aupr =                 row[8]
        self.fpr_at_tpr =           row[9]
        self.detection_error =      row[10]
        self.max_iou =              row[11]
        self.had_error =            row[12]
    
    def __getitem__(self, key):
        return self.__dict__[key]
    
    def __str__(self):
        return "<{} {} {} {}>".format(self.processor_type,
                                    self.data_config.replace("configs/data/", "").replace(".config", ""),
                                    self.annot_type,
                                    self.arg_group_id)

    def __repr__(self):
        return self.__str__()

def get_result_no_train(result_list):
    cur_best = {
        "auroc": None,
        "aupr": None,
        "fpr_at_tpr": None,
        "detection_error": None,
        "max_iou": None,
    }

    for r in result_list:
        if cur_best["auroc"] is None or cur_best["auroc"].auroc < r.auroc:
            cur_best["auroc"] = r
        
        if cur_best["aupr"] is None or cur_best["aupr"].aupr < r.aupr:
            cur_best["aupr"] = r
        
        if cur_best["fpr_at_tpr"] is None or cur_best["fpr_at_tpr"].fpr_at_tpr > r.fpr_at_tpr:
            cur_best["fpr_at_tpr"] = r
        
        if cur_best["detection_error"] is None or cur_best["detection_error"].detection_error > r.detection_error:
            cur_best["detection_error"] = r
        
        if cur_best["max_iou"] is None or cur_best["max_iou"].max_iou < r.max_iou:
            cur_best["max_iou"] = r
    
    return cur_best["auroc"], cur_best["aupr"], cur_best["fpr_at_tpr"], cur_best["detection_error"], cur_best["max_iou"]

def update_best(cur, cur_best, metric, cmp, eval_arg_ids):
    if cur_best[metric] is None:
        cur_best[metric] = cur
    elif cur_best[metric][metric] == cur[metric] and cur.arg_group_id in eval_arg_ids:
        cur_best[metric] = cur
    elif cmp(cur_best[metric][metric], cur[metric]):
        cur_best[metric] = cur

def get_result_train(train_result, eval_result):

    cur_best = {
        "auroc": None,
        "aupr": None,
        "fpr_at_tpr": None,
        "detection_error": None,
        "max_iou": None,
    }

    eval_arg_ids = set([r.arg_group_id for r in eval_result])

    lt = lambda x,y: x < y
    gt = lambda x,y: x > y

    for r in train_result:
        update_best(r, cur_best, "auroc",           lt, eval_arg_ids)
        update_best(r, cur_best, "aupr",            lt, eval_arg_ids)
        update_best(r, cur_best, "fpr_at_tpr",      gt, eval_arg_ids)
        update_best(r, cur_best, "detection_error", gt, eval_arg_ids)
        update_best(r, cur_best, "max_iou",         lt, eval_arg_ids)
    
    best_train = cur_best["auroc"], cur_best["aupr"], cur_best["fpr_at_tpr"], cur_best["detection_error"], cur_best["max_iou"]

    # best_train = get_result_no_train(train_result)

    result = [None,None,None,None,None]

    for r in eval_result:
        for i, bt in enumerate(best_train):
            if r.arg_group_id == bt.arg_group_id:
                result[i] = r
    
    if None in result:
        import pdb; pdb.set_trace()
        print("something bad")

    return tuple(result)

def format_row(dataset, p_type, values, model_name):
    values = ["{:0.5f}".format(v) for v in values]
    vals = "\t& ".join(values)
    ret = "\t{}\t& {}".format(dataset.replace("_eval",""), vals) + " \\\\"
    ret = ret.replace("_", "\\_")
    return model_name, p_type, ret

def main():
    create = """CREATE VIEW results_view as
                select ec.model_config, ec.data_config, ec.trained_checkpoint, ec.processor_type, ec.annot_type, e.id as experiment_id, e.arg_group_id, r.auroc, r.aupr, r.fpr_at_tpr, r.detection_error, r.max_iou, r.had_error
                from experimentconfig as ec, experiment as e, result as r
                where ec.id == e.config_id
                and e.id == r.experiment_id"""

    all_views = set([v.name for v in db.db.get_views()])

    if "results_view" not in all_views:
        db.db.execute_sql(create)

    results = db.db.execute_sql("select * from results_view where had_error == 0")

    to_group = ["model_config", "trained_checkpoint", "processor_type", "annot_type"]

    results = [ResultWrapper(row) for row in results]

    groups = {}
    for res in results:
        if res.annot_type == "error":
            continue
        g = tuple([res[i] for i in to_group])

        data = res.data_config

        if g not in groups:
            groups[g] = {data: [res]}
        else:
            if data not in groups[g]:
                groups[g][data] = [res]
            else:
                groups[g][data].append(res)
    
    all_results = []

    chosen_args = {}

    for g in groups:
        for data in groups[g]:
            data_print = data
            if "train" in data:
                eval_data = data.replace("train", "eval")
                if eval_data in groups[g]:
                    cur_result = get_result_train(groups[g][data], groups[g][eval_data])
                    data_print = eval_data
                else:
                    print("train with no eval", data)
                    cur_result = get_result_no_train(groups[g][data])
            elif "eval" in data:
                train_data = data.replace("eval", "train")
                if train_data in groups[g]:
                    #eval data is processed when train data is encountered
                    continue
                else:
                    #no train
                    cur_result = get_result_no_train(groups[g][data])
            else:
                cur_result = get_result_no_train(groups[g][data])
            # import pdb; pdb.set_trace()
            model_name = g[0].replace("configs/model/", "").replace(".config", "")
            order = ["auroc", "aupr", "fpr_at_tpr", "detection_error", "max_iou"]
            to_print = [res[f] for f, res in zip(order, cur_result)]
            selected_kwargs = db.KeyWordArgs.select().where((db.KeyWordArgs.group_id == cur_result[0].arg_group_id))
            chosen_args[g[2]] = [(kw.name, kw.value) for kw in selected_kwargs]
            
            all_results.append((data_print.replace("configs/data/", "").replace(".config", ""), g[2], to_print, model_name))
    
    print("Selected Args:")
    pprint(chosen_args)
    # import pdb; pdb.set_trace()
    second_sort = {"sun_eval":          0, "sun_train":         0, 
                    "coco_eval":        1, "coco_train":        1,
                    "coco_city_eval":   2, "coco_city_train":   2,
                    "idd_cars_eval":    3, "idd_cars_train":    3,
                    "lostfound_eval":   4, "lostfound_train":   4,
                    "uniform_eval":     5, "uniform_train":     5,
                    "normal_eval":      6, "normal_train":      6,
                    "perlin_eval":      7, "perlin_train":      7}
    first_sort = {"MaxSoftmax": 0, "ODIN": 1, "Mahal": 2, "Confidence": 3, "Dropout": 4, "Entropy": 5, "AlEnt": 6}
    all_results = sorted(all_results, key= lambda x: (first_sort[x[1]], int("pspnet" in x[3]), second_sort[x[0]]))

    format_results = list(map(lambda x: format_row(*x), all_results))
    
    start = "\pgfplotstableread[row sep=\\\\,col sep=&]{\n" \
                "\tDataset & AUROC     & AUPRC     & FPRatTPR  & DE        & MaxIoU \\\\"
    # print("auroc, aupr, fpr_at_tpr, detection_error, max_iou")
    prev_model_name = None
    prev_p_type = None
    for model_name, p_type, res in format_results:
        if prev_model_name is None or model_name != prev_model_name or p_type != prev_p_type:
            end = "}\\" + str(prev_model_name) + str(prev_p_type) + "data\n"
            print(end)
            print(start)
            prev_model_name = model_name
            prev_p_type = p_type
        print(res)

    end = "}\\" + model_name + p_type + "data\n"
    print(end)