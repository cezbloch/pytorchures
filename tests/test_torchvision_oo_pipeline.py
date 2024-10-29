from torchvision.io.image import read_image
from torchvision.models.detection import (
    RetinaNet_ResNet50_FPN_Weights,
    retinanet_resnet50_fpn,
)

from pytorchures import TimedLayer
from pytorchures.torchvision_pipeline import TorchVisionObjectDetectionPipeline

IMAGE_PATH = "data/images/traffic.jpg"


def test_torchvision_oo_pipeline_can_execute_with_wrapped_model():
    weights = RetinaNet_ResNet50_FPN_Weights.DEFAULT
    model = retinanet_resnet50_fpn(weights=weights, box_score_thresh=0.9)
    model = TimedLayer(model)
    model.eval()
    preprocess = weights.transforms()
    device = "cpu"
    image = read_image(IMAGE_PATH)
    categories = weights.meta["categories"]

    pipeline = TorchVisionObjectDetectionPipeline(model=model, preprocessor=preprocess, categories=categories, device=device)

    input_tensor = pipeline.preprocess(image)
    output_tensor = pipeline.predict(input_tensor)
    image_with_boxes = pipeline.postprocess(output_tensor)

    assert image_with_boxes.size[0] == image.shape[2]
    assert image_with_boxes.size[1] == image.shape[1]
