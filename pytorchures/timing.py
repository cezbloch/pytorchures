import logging
import time
from functools import wraps
from typing import Dict

import numpy as np
import torch

logging.basicConfig(
    filename="profiling.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class AcceleratorSynchronizer:
    def __init__(self, device_type: str) -> None:
        self._device_type = device_type

        if self._device_type == "cpu":
            self._synchronize = torch.cpu.synchronize
        elif self._device_type == "cuda":
            self._synchronize = torch.cuda.synchronize
        elif self._device_type == "xpu":
            self._synchronize = torch.xpu.synchronize
        elif self._device_type == "mtia":
            self._synchronize = torch.mtia.synchronize
        elif self._device_type is None:
            self._synchronize = lambda: None
        else:
            raise ValueError(f"Device type '{self._device_type}' is not supported.")

    def __call__(self) -> None:
        self._synchronize()


class TimedModule(torch.nn.Module):
    """A wrapper class to measure the time taken by a layer in milliseconds"""

    def __init__(self, module: torch.nn.Module, indent: str = "\t"):
        super().__init__()
        assert isinstance(module, torch.nn.Module)
        assert not isinstance(module, TimedModule)
        self._module = module
        self._module_name = module.__class__.__name__
        self._execution_times_ms: list[float] = []
        self._indent = indent

        wrap_model_layers(module, indent + "\t")

    def forward(self, *args, **kwargs):
        with torch.no_grad():
            start_time = time.time()
            x = self._module(*args, **kwargs)
            AcceleratorSynchronizer(self.get_device_type())()
            end_time = time.time()
            execution_time_ms = (end_time - start_time) * 1000
            self._execution_times_ms.append(execution_time_ms)
            return x

    def get_device_type(self):
        params = self._module.parameters()
        device_type = None
        for i in params:
            device_type = i.device.type

        return device_type

    def __len__(self):
        return len(self._module)

    def __iter__(self):
        return iter(self._module)

    def __getattr__(self, attribute_name):
        """
        Delegate all other attribute access to the wrapped layer.

        NOTE: __getattr__ is called when the attribute is not found in the object's dictionary.
        """
        try:
            return super().__getattr__(attribute_name)
        except AttributeError:
            return getattr(self._module, attribute_name)

    def get_timings(self) -> Dict:
        profiling_data = {
            "module_name": self._module_name,
            "device_type": self.get_device_type(),
        }

        if len(self._execution_times_ms) > 0:
            exec_times = {
                "execution_times_ms": self._execution_times_ms,
                "mean_time_ms": np.mean(self._execution_times_ms),
                "median_time_ms": np.median(self._execution_times_ms),
            }

            profiling_data.update(exec_times)

        children = []

        for _, child in self._module.named_children():
            if isinstance(child, TimedModule):
                children.append(child.get_timings())

        if len(children) > 0:
            profiling_data["sub_modules"] = children

        return profiling_data


def wrap_model_layers(model, indent="\t") -> None:
    """Wrap all torch Module layers of a given model with TimedLayer, to print each layer execution time."""
    assert isinstance(model, torch.nn.Module)
    assert not isinstance(model, TimedModule)

    print(f"{indent}{model.__class__.__name__}")

    for attribute_name, child in model.named_children():
        wrapped_child = TimedModule(child, indent)
        setattr(model, attribute_name, wrapped_child)


def profile_function(f):
    """Decorator to profile function calls. Prints the time taken by the function in milliseconds."""

    @wraps(f)
    def wrap(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        elapsed_ms = (te - ts) * 1000
        logger.info(f"Function '{f.__name__}' executed in {elapsed_ms:.4f} ms.")
        return result

    return wrap
