import consts

from multiprocessing import Process

def _run_task_with_gpu(task, gpu):
    # consider approach
    # import os
    # os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"
    import theano.sandbox.cuda
    theano.sandbox.cuda.use(gpu)
    task.run()

class TaskManager(object):
    def __init__(self, tasks):
        self.tasks = tasks

    def run_tasks(self):
        if len(self.tasks) > consts.num_gpus:
            raise Exception('Does not support that many gpus')

        processes = []
        for index, task in enumerate(self.tasks):
            p = Process(
                    target=_run_task_with_gpu,
                    args=(task, 'gpu' + str(index))
                    )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
