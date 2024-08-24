
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
        self._total_time = (end_time - start_time) * 1000
        print(f"Layer {self.layer.__class__.__name__}: {self._total_time:.6f} ms")
        return x
    
    def postprocess(self, *args, **kwargs):
        return self.layer.postprocess(*args, **kwargs)
    
    def __len__(self):
        return len(self.layer)
    
    def __iter__(self):
        return iter(self.layer)
    
    def __next__(self):
        return next(iter(self.layer))    
    
    # def __getattribute__(self, name):
    #     print(f"Intercepted access to: {name}")
    #     try:
    #         print(f"Trying to get attribute: {name}")
    #         return TimedLayer().__getattribute__(name)
    #     except TypeError as e:
    #         return self.layer.__getattribute__(name)            
    #     except Exception as e:
    #         print(f"Error: {e}")
    #         def method(*args, **kwargs):
    #             print(f"Handling non-existent method: {name}")
    #             return f"Handled method: {name}"
    #         return method
        

    def get_time(self):
        return self._total_time


def wrap_model_layers(model):
    attributes = dir(model)
    for a in attributes:
        attr = getattr(model, a)
        if isinstance(attr, torch.nn.Module):
            if not isinstance(model, torch.nn.Sequential):
                setattr(model, a, TimedLayer(attr))
            wrap_model_layers(attr)


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
