from typing import Callable, List, Tuple

import torch
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_bounding_boxes

from pytorchures.timing import profile_function


class TorchVisionObjectDetectionPipeline:
    def __init__(
        self,
        model: torch.nn.Module,
        preprocessor: Callable,
        categories: List,
        device: str,
    ):
        self.model = model.to(device)
        self.preprocessor = preprocessor
        self.categories = categories
        self.timings = {}
        self.device = device
        self.input_image = None

    @profile_function
    def preprocess(self, image):
        self.image = image.to(self.device)
        assert self.image.device.type == self.device
        x = self.preprocessor(self.image)
        assert x.device.type == self.device
        return [x]

    @profile_function
    def predict(self, inputs):
        assert next(self.model.parameters()).device.type == self.device
        outputs = self.model(inputs)
        return outputs[0]

    @profile_function
    def postprocess(self, predictions):
        boxes = predictions["boxes"]
        assert boxes.device.type == self.device

        labels = [self.categories[i] for i in predictions["labels"]]

        box = draw_bounding_boxes(
            self.image,
            boxes=boxes,
            labels=labels,
            colors="red",
            width=4,
            font_size=30,
        )

        image_with_boxes = to_pil_image(box.detach())
        return image_with_boxes
