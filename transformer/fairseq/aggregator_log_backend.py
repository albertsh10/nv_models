from collections import OrderedDict
import time
from dllogger import Backend

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.updated = False
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value):
        self.updated = True
        if isinstance(value, tuple) or isinstance(value, list):
            val = value[0]
            n = value[1]
        else:
            val = value
            n = 1
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    @property
    def value(self):
        return self.avg

class PerformanceMeter(object):
    def __init__(self):
        self.reset() 

    def reset(self):
        self.updated = False
        self.start = time.time()
        self.n = 0

    def update(self, val=1):
        self.updated = True
        self.n += val

    @property
    def value(self):
        return self.n / self.elapsed_time

    @property
    def elapsed_time(self):
        return time.time() - self.start

METRIC = {'average':AverageMeter, 'performance':PerformanceMeter}

class AggregatorBackend(Backend):
    def __init__(self, verbosity, agg_dict):
        super().__init__(verbosity=verbosity)
        agg_dict = OrderedDict({k: ((v,) if not(isinstance(v,tuple) or isinstance(v, list)) else v) for k,v in agg_dict.items()})
        self.metrics = OrderedDict({k: [METRIC[x]() for x in v] for  k,v in agg_dict.items()})
        self.metrics.flushed = True
        self.step = 0
        self.epoch = 0
        self.start_time = time.time()
    
    @property
    def log_level(self):
        return self._log_level

    def metadata(self, timestamp, elapsedtime, metric, metadata):
        pass

    def _reset_perf_meter(self, name):
        for agg in self.metrics[name]:
            if isinstance(agg, PerformanceMeter):
                agg.reset()
    def log(self, timestamp, elapsedtime, step, data):
        self.step = step
        if 'epoch' in data.keys():
            self.epoch = data['epoch']
        for k, v in data.items():
            if k not in self.metrics.keys():
                continue
            self.metrics.flushed = False
            for ag in self.metrics[k]:
                ag.update(v)

    def flush(self):
        if self.metrics.flushed:
            return
        result_string = 'Transformer | epoch {} | step {} |'.format(self.epoch, self.step)
        for name, aggregators in self.metrics.items():
            for agg in aggregators:
                if not agg.updated:
                    continue
                if isinstance(agg, AverageMeter):
                    _name = 'avg ' + name
                elif isinstance(agg, PerformanceMeter):
                    _name = name + '/s'

                result_string += _name + ' {:.3f} |'.format(agg.value)
                agg.reset()
        
        result_string += 'walltime {:.3f} |'.format(time.time() - self.start_time)
        self.metrics.flushed = True
        print(result_string)

