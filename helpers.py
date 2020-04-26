import time
from os import path, makedirs


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name,)
        print('Elapsed: %s' % (time.time() - self.tstart))

    def getElapsed(self):
        return time.time() - self.tstart


def createFoldersForModel(model_name, dataset):
    cache_path = 'cache/' + dataset
    if not path.exists(cache_path):
        makedirs(cache_path)

    results_path = 'results/' + model_name + '/' + dataset
    if not path.exists(results_path):
        makedirs(results_path)
