import argparse
import logging

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.models import get_model, get_model_weights, list_models

from pytorchures.timing import wrap_model_layers
from pytorchures.torchvision_pipeline import \
    TorchVisionObjectDetectionPipeline

LOG_FILENAME = "profiling.log"
logging.basicConfig(
    filename=LOG_FILENAME,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main(device: str, nr_images: int, show_image: bool, model_name: str):
    logger.info(f"Saving logs to {LOG_FILENAME}")

    if device == "cuda":
        if torch.cuda.is_available():
            logger.info("Executing on GPU")
        else:
            msg = "CUDA is not available."
            logger.error(msg)
            raise ValueError(msg)
    else:
        logger.info("Executing on CPU")

    transform = T.Compose(
        [
            T.ToTensor(),
        ]
    )

    voc_dataset = datasets.VOCDetection(
        root=".data", year="2007", image_set="val", download=True, transform=transform
    )
    data_loader = DataLoader(
        dataset=voc_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=1,
        collate_fn=lambda x: tuple(zip(*x)),
    )

    logger.info(f"Fetching pretrained model '{model_name}'.")
    weights = get_model_weights(model_name).DEFAULT
    model = get_model(model_name, weights="DEFAULT")
    model.eval()

    wrap_model_layers(model)
    preprocess = weights.transforms()
    wrap_model_layers(preprocess)
    categories = weights.meta["categories"]

    pipeline = TorchVisionObjectDetectionPipeline(
        model=model, preprocessor=preprocess, categories=categories, device=device
    )
    image_count = 0

    for batch_images, _ in data_loader:
        msg = f"----------------Processing image {image_count + 1} -----------------"
        logger.info(msg)
        print(msg)  # I know this can be done smarter...

        for i in range(len(batch_images[:nr_images])):
            image = batch_images[i]

            input_tensor = pipeline.preprocess(image)
            output_tensor = pipeline.predict(input_tensor)
            image_with_boxes = pipeline.postprocess(output_tensor)
            if show_image:
                image_with_boxes.show()

            image_count += 1

        if image_count >= nr_images:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Specify the device to run the model on - valid options are ['cpu', 'cuda'].",
    )
    parser.add_argument(
        "--nr_images",
        type=int,
        default=2,
        help="Select how many images should be processed from the dataset.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="retinanet_resnet50_fpn",
        help=f"Select the model to use from the list of available models: {list_models(module=torchvision.models.detection)}",
    )
    parser.add_argument(
        "--show_image",
        action="store_true",
        help="Flag to determine whether to display the image with detected boxes.",
    )
    args = parser.parse_args()

    main(args.device, args.nr_images, args.show_image, args.model_name)
