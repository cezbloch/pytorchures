import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import FrozenBatchNorm2d
from object_detection.timing import wrap_model_layers, TimedLayer


def test_conv_layer_wrapping():
    model = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
    model = TimedLayer(model)
    input_tensor = torch.randn(1, 1, 28, 28)

    output = model(input_tensor)

    assert output is not None
    assert model.get_time() > 0


def test_sequential_wrapping():
    model = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
    )

    wrap_model_layers(model)

    layers = model.named_children()

    _, timed_conv = next(layers)

    assert isinstance(timed_conv, TimedLayer)
    assert isinstance(timed_conv.layer, nn.Conv2d)


def test_nested_sequential_wrapping():
    model = nn.Sequential(
        nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
        ),
        nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
    )

    wrap_model_layers(model)

    layers = model.named_children()

    _, timed_seq = next(layers)
    _, nested_conv = next(timed_seq.layer.named_children())


    assert isinstance(nested_conv, TimedLayer)
    assert isinstance(nested_conv.layer, nn.Conv2d)


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1
        )
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


def test_named_model_fields_are_wrapped():
    model = SimpleCNN()
    input_tensor = torch.randn(1, 1, 28, 28)
    wrap_model_layers(model)

    output = model(input_tensor)

    assert isinstance(model.conv1, TimedLayer)
    assert output is not None
    


class MyCustomLayer(nn.Conv2d):
    """This class is to demonstrate call on a non-existent method"""

    def custom_method(self):
        pass


class SimpleCNNWithCustomMethodCall(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = MyCustomLayer(
            in_channels=1, out_channels=16, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1
        )
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # when wrapped with TimedLayer, this will raise an error if not handled properly
        self.conv1.custom_method()

        return x


def test_forward_call_of_non_existent_method():
    """
    This tests demonstrates a problem with TimedLayer.
    Currently, each method that in called on an object being wrapped needs to be exposed in TimedLayer.
    This is not ideal as it may require TimedLayer to be extended for new models.
    In the future, we may try to intercept all method calls on the object and forward them to the layer being wrapped.
    """
    model = SimpleCNNWithCustomMethodCall()
    input_tensor = torch.randn(1, 1, 28, 28)
    wrap_model_layers(model)

    with pytest.raises(AttributeError):
        _ = model(input_tensor)
