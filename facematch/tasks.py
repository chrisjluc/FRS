
class GPUTask(object):

    def run(self):
        raise NotImplementedError

class TrainingTask(GPUTask):

    def __init__(self, model_cls, X_train, Y_train, user_ids, model_name):
        self.model_cls = model_cls
        self.X_train = X_train
        self.Y_train = Y_train
        self.user_ids = user_ids
        self.model_name = model_name

    def run(self):
        model = self.model_cls(self.X_train, self.Y_train, self.user_ids, self.model_name)
        model.train()
        model.save()
