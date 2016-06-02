
class GPUTask(object):

    def run(self):
        raise NotImplementedError

class TrainingTask(GPUTask):

    def __init__(self, model_cls, images, user_ids, model_name):
        self.model_cls = model_cls
        self.images = images
        self.user_ids = user_ids
        self.model_name = model_name

    def run(self):
        model = self.model_cls(self.images, self.user_ids, self.model_name)
        model.train()
        model.save()

class ScoringTask(GPUTask):

    def run(self):
        pass
