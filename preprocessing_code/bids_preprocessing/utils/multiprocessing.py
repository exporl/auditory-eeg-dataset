import logging

import multiprocessing



class SimpleCallbackFn:

    def __init__(self):
        self.total = None
        self.current_count = 0

    def __call__(self, x):
        self.current_count += 1
        logging.info(f"Progress {self.current_count}/{self.total}.")

class MultiprocessingSingleton:
    manager = multiprocessing.Manager()

    to_clean = []

    @classmethod
    def get_map_fn(cls, nb_processes, callback=SimpleCallbackFn()):
        if nb_processes != 1:

            if nb_processes == -1:
                nb_processes = multiprocessing.cpu_count()
            pool = multiprocessing.Pool(nb_processes)
            cls.to_clean += [pool]

            def dummy_map_fn(fn, iterable):
                callback.total = len(iterable)
                return [pool.apply_async(fn, args=[item], callback=callback) for item in iterable]
            map_fn = dummy_map_fn
        else:
            map_fn = map
        return map_fn

    @classmethod
    def clean(cls):
        for pool in cls.to_clean:
            pool.close()
            pool.join()
        cls.to_clean = []
