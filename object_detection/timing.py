
import time
from functools import wraps

import torch


class TimedLayer(torch.nn.Module):
    def __init__(self, layer):
        super(TimedLayer, self).__init__()
        self.layer = layer
        self._total_time = 0.0

    def forward(self, *args, **kwargs):
        start_time = time.time()
        x = self.layer(*args, **kwargs)
        torch.cuda.synchronize()  # Synchronize if using GPU
        end_time = time.time()
        self._total_time = end_time - start_time
        print(f"Layer {self.layer.__class__.__name__}: {self._total_time:.6f} seconds")
        return x
    
    def postprocess(self, *args, **kwargs):
        return self.layer.postprocess(*args, **kwargs)

    def get_time(self):
        return self._total_time


def wrap_model_layers(model):
    #l = [module for module in model.modules() if not isinstance(module, torch.nn.Sequential)]
    attributes = dir(model)
    for a in attributes:
        if isinstance(getattr(model, a), torch.nn.Module):
            setattr(model, a, TimedLayer(getattr(model, a)))


def profile_function(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        elapsed_time = te - ts
        print(f"Function '{f.__name__}' executed in {elapsed_time:.4f} seconds")
        return result

    return wrap
