
import time
from functools import wraps
from time import time

import torch


class TimedLayer(torch.nn.Module):
    def __init__(self, layer):
        super(TimedLayer, self).__init__()
        self.layer = layer
        self._total_time = 0.0

    def forward(self, x):
        start_time = time.time()
        x = self.layer(x)
        torch.cuda.synchronize()  # Synchronize if using GPU
        end_time = time.time()
        self._total_time = end_time - start_time
        print(f"Layer {self.layer.__class__.__name__}: {self._total_time:.6f} seconds")
        return x

    def get_time(self):
        return self._total_time


def wrap_model_layers(model):
    for name, module in model.named_children():
        print(f"Wrapping layer {name}")
        module = TimedLayer(module)


def profile_function(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        elapsed_time = te - ts
        print(f"Function '{f.__name__}' executed in {elapsed_time:.4f} seconds")
        return result

    return wrap
