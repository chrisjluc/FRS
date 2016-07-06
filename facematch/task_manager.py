import consts

from multiprocessing import Process
from threading import Thread, BoundedSemaphore, Lock

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
        num_gpus = consts.num_gpus
        self.gpu_locks = tuple([Lock() for _ in range(num_gpus)])
        self.lock_for_gpu_lock = Lock()
        self.semaphore = BoundedSemaphore(num_gpus)

    def run_tasks(self):
        threads = []
        for task in self.tasks:
            t = Thread(target=self._run_task, args=(task,))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

    def _run_task(self, task):
        self.semaphore.acquire()
#        print 'Semaphore acquired'
        gpu_lock_index = None
        gpu_lock = None

        self.lock_for_gpu_lock.acquire()
        for index, lock in enumerate(self.gpu_locks):
            if not lock.locked():
                gpu_lock = lock
                gpu_lock_index = index
                break
        if not gpu_lock:
            raise Exception('Unable to obtain a gpu lock')
        if gpu_lock_index is None:
            raise Exception('Unable to obtain a gpu lock index')

        gpu_lock.acquire()
#        print 'GPU acquired: ' + str(gpu_lock_index)
        self.lock_for_gpu_lock.release()

        p = Process(
                target=_run_task_with_gpu,
                args=(task, 'gpu' + str(gpu_lock_index))
                )
        p.start()
        p.join()

        gpu_lock.release()
#        print 'GPU released: ' + str(gpu_lock_index)
        self.semaphore.release()
#        print 'Semaphore released'

