import json_tricks as json
import numpy as np

class Experiment(object):

    def __init__(self, model_config, data_config,
                trained_checkpoint, pad_to_shape,
                processor_type, annot_type, kwargs):
        
        self.model_config = model_config
        self.data_config = data_config
        self.trained_checkpoint = trained_checkpoint
        self.pad_to_shape = pad_to_shape
        self.processor_type = processor_type
        self.annot_type = annot_type
        self.kwargs = kwargs
    
    def serialze(self):
        return json.dumps(self.__dict__)
    
    @staticmethod
    def deserialze(j):
        data = json.loads(j, cls_lookup_map={"__ndarray__": np.ndarray})
        return Experiment(**data)

class Result(object):

    def __init__(self, buff, result, had_error):
        self.buff = buff
        self.result = result
        self.had_error = had_error

    def serialze(self):
        return json.dumps(self.__dict__)
    
    @staticmethod
    def deserialze(j):
        data = json.loads(j, cls_lookup_map={"__ndarray__": np.ndarray})
        return Result(**data)

class ExperimentDataset(object):

    def __init__(self, train_set, eval_set=None):
        self.train_set = train_set
        self.eval_set = eval_set
        if eval_set is None:
            self.eval_set = train_set