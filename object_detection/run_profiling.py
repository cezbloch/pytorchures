import argparse

import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.models.detection import (RetinaNet_ResNet50_FPN_Weights,
                                          retinanet_resnet50_fpn)

from object_detection.timing import wrap_model_layers
from object_detection.torchvision_pipeline import \
    TorchVisionObjectDetectionPipeline


def main(device: str, nr_images: int, show_image: bool):
    transform = T.Compose(
        [
            T.ToTensor(),  # Converts the PIL image to a PyTorch tensor
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

    weights = RetinaNet_ResNet50_FPN_Weights.DEFAULT
    model = retinanet_resnet50_fpn(weights=weights, box_score_thresh=0.95)
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
        print(f"----------------Processing image {image_count + 1} -----------------")
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
        help="Specify the device to run the model on.",
    )
    parser.add_argument(
        "--nr_images",
        type=int,
        default=2,
        help="Select how many images should be processed from the dataset.",
    )
    parser.add_argument(
        "--show_image",
        action="store_true",
        help="Flag to determine whether to display the image with detected boxes.",
    )
    args = parser.parse_args()

    main(args.device, args.nr_images, args.show_image)
