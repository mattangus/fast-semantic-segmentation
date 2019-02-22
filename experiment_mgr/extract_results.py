from . import db
from . import db_helper

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
had_error = 10

def get_result_no_train(result_list):
    cur_best = {auroc: [-1, 0.], aupr: [-1, 0.], fpr_at_tpr: [-1, 1.], detection_error: [-1, 1.]}

    for r in result_list:
        if cur_best[auroc][1] < r[1]:
            cur_best[auroc][0] = r[0]
            cur_best[auroc][1] = r[1]

        if cur_best[aupr][1] < r[2]:
            cur_best[aupr][0] = r[0]
            cur_best[aupr][1] = r[2]
            

        if cur_best[fpr_at_tpr][1] > r[3]:
            cur_best[fpr_at_tpr][0] = r[0]
            cur_best[fpr_at_tpr][1] = r[3]

        if cur_best[detection_error][1] > r[4]:
            cur_best[detection_error][0] = r[0]
            cur_best[detection_error][1] = r[4]
    
    return cur_best[auroc], cur_best[aupr], cur_best[fpr_at_tpr], cur_best[detection_error]

def get_result_train(train_result, eval_result):
    
    best_train = get_result_no_train(train_result)

    best_arg = [bt[0] for bt in best_train]

    result = [[],[],[],[]]

    for r in eval_result:
        for i, ba in enumerate(best_arg):
            if r[0] == ba:
                result[i] = [ba, r[i]]

    return tuple(result)

def main():
    results = db.db.execute_sql("select * from results_view where had_error == 0")

    to_group = [model_config, trained_checkpoint, processor_type, annot_type]
    result_items = [arg_group_id, auroc, aupr, fpr_at_tpr, detection_error]

    groups = {}
    for row in results:
        g = tuple([row[i] for i in to_group])
        r = tuple([row[i] for i in result_items])

        data = row[data_config]       

        if g not in groups:
            groups[g] = {data: [r]}
        else:
            if data not in groups[g]:
                groups[g][data] = [r]
            else:
                groups[g][data].append(r)
        
    for g in groups:
        for data in groups[g]:
            if "train" in data:
                eval_data = data.replace("train", "eval")
                if eval_data in groups[g]:
                    cur_result = get_result_train(groups[g][data], groups[g][eval_data])
                else:
                    print("train with no eval")
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
            

            print(g, data, cur_result)

    import pdb; pdb.set_trace()
    print("here")