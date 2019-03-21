
from abc import abstractmethod

class PostProcessor(object):
    
    def __init__(self, name, model, outputs_dict, num_gpus):
        """Create a post processor.
           This is an abstract class that should not be instanciated
        
        Arguments:
            name {[str]} -- name of post processor
            model {[PSPNetArchitecture]} -- pspnet model
            outputs_dict {[dict]} -- output dict from pspnet model
        """
        self.name = name
        self.model = model
        self.outputs_dict = outputs_dict
        self.num_gpus = num_gpus
    
    @abstractmethod
    def get_preprocessed(self):
        """Get preprocessed input if needed. Returns None otherwise.
        """
        pass

    @abstractmethod
    def get_init_feed(self):
        """get an init feed dict to merge with master dict
        """
        pass

    @abstractmethod
    def get_vars_noload(self):
        """Get list of variables to not load from checkpoint
        """
        pass

    @abstractmethod
    def post_process_ops(self):
        """Create any tf graph ops required for \
           post processing
        """
        pass

    @abstractmethod
    def post_process(self, numpy_dict):
        """ apply and track any post processing with \
            numpy array output
        
        Arguments:
            numpy_dict {[dict]} -- outputs from sess run
        """
        pass
    
    @abstractmethod
    def get_fetch_dict(self):
        """ get fetch dictionary to merge with master \
            fetch dict
        """
        pass
    
    @abstractmethod
    def get_feed_dict(self):
        """get feed dict to merge with master feed dict
        """
        pass
    
    @abstractmethod
    def get_output_image(self):
        """ get the output ood heat map
        """
        pass

    @abstractmethod
    def get_prediction(self):
        """ get the prediction map (cityscapes classes)
        """
        pass
    
    @abstractmethod
    def get_weights(self):
        """ get the weights used for metrics
        """
        pass


#TODO Delete this example:
# from . import post_processor as pp
# from third_party.doc_inherit import doc_inherit

# class MahalProcessor(pp.PostProcessor):
    
#     def __init__(self, name, model, outputs_dict):
#         super().__init__("mahal", model, outputs_dict)
    
#     @doc_inherit
#     def get_init_feed(self):
#         pass

#     @doc_inherit
#     def get_vars_noload(self):
#         pass

#     @doc_inherit
#     def post_proces_ops(self):
#         pass

#     @doc_inherit
#     def post_process(self, numpy_dict):
#         pass
    
#     @doc_inherit
#     def get_fetch_dict(self):
#         pass
    
#     @doc_inherit
#     def get_feed_dict(self):
#         pass
    