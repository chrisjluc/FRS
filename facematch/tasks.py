
class Task(object):

    def run(self):
        raise NotImplementedError

class TrainingTask(Task):

    def run(self):
        pass

class ScoringTask(Task):

    def run(self):
        pass
