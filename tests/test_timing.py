import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorchures import TimedLayer, wrap_model_layers


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
    assert isinstance(timed_conv._module, nn.Conv2d)


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
    _, nested_conv = next(timed_seq._module.named_children())

    assert isinstance(nested_conv, TimedLayer)
    assert isinstance(nested_conv._module, nn.Conv2d)


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
    assert isinstance(model.conv2, TimedLayer)
    assert isinstance(model.fc1, TimedLayer)
    assert isinstance(model.fc2, TimedLayer)
    assert output is not None


def test_model_sublayer_timings_are_retrieved():
    model = SimpleCNN()
    input_tensor = torch.randn(1, 1, 28, 28)
    timed_model = wrap_model_layers(model)

    _ = timed_model(input_tensor)
    timings_dict = timed_model.get_timings()

    assert isinstance(timed_model, TimedLayer)
    assert len(timings_dict) == 3
    assert len(timings_dict["sub_modules"]) == 4
    assert timings_dict["module_name"] == "SimpleCNN"
    assert timings_dict["sub_modules"][0]["module_name"] == "Conv2d"
    assert timings_dict["sub_modules"][0]["sub_modules"] == []
    assert timings_dict["sub_modules"][2]["module_name"] == "Linear"


def test_3_time_measurements_are_available_when_model_is_called_3_times():
    model = SimpleCNN()
    input_tensor = torch.randn(1, 1, 28, 28)
    timed_model = wrap_model_layers(model)

    _ = timed_model(input_tensor)
    _ = timed_model(input_tensor)
    _ = timed_model(input_tensor)
    timings_dict = timed_model.get_timings()

    assert len(timings_dict["execution_times_ms"]) == 3
    for i in range(4):
        assert len(timings_dict["sub_modules"][i]["execution_times_ms"]) == 3


class MyCustomLayer(nn.Conv2d):
    """This class is to demonstrate call on a non-existent method"""

    def custom_method(self):
        self.called = True


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


def test_method_of_wrapped_layer_can_be_accessed_through_timed_layer():
    conv = MyCustomLayer(in_channels=1, out_channels=2, kernel_size=3)
    timed_conv = TimedLayer(conv)

    method = timed_conv._module.custom_method
    forwarded_method = timed_conv.custom_method

    assert method == forwarded_method


def test_method_of_wrapper_layer_is_called_during_model_execution():
    model = SimpleCNNWithCustomMethodCall()
    input_tensor = torch.randn(1, 1, 28, 28)
    wrap_model_layers(model)

    _ = model(input_tensor)

    assert isinstance(model.conv1, TimedLayer)
    assert model.conv1.called is True
