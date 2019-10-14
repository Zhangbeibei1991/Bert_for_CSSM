class InputExample(object):
    '''a single train/test example for simple sequence classification'''
    def __init__(self,guid,text_a,text_b=None,label=None):
        '''construct an InputExample, I guess here that we can input some external features'''
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    def __init__(self,input_ids,input_mask, segment_ids, label_id):
        """A single set of features of data."""
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""
    def get_train_examples(self,file_dir):
        raise NotImplementedError()

    def get_dev_examples(self,file_dir):
        raise NotImplementedError()

    def get_test_examples(self,file_dir):
        raise NotImplementedError()

    def get_labels(self):
        raise NotImplementedError()

class TriggerProcessor(DataProcessor):
    def get_train_examples(self,file_dir):
        pass
