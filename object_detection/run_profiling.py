import argparse

import torchvision
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.io.image import read_image

from object_detection.torchvision_pipeline import TorchVisionObjectDetectionPipeline

from torchvision.models.detection import (RetinaNet_ResNet50_FPN_Weights,
                                          retinanet_resnet50_fpn)

from object_detection.timing import wrap_model_layers


def main(image_path: str, model_url: str):
    #image = Image.open(image_path).convert("RGB")
    #image = read_image("data/images/traffic.jpg")
    
    transform = T.Compose([
        T.ToTensor(),  # Converts the PIL image to a PyTorch tensor
    ])

    voc_dataset = datasets.VOCDetection(root='.data', year='2007', image_set='val', download=True, transform=transform)
    data_loader = DataLoader(dataset=voc_dataset, batch_size=2, shuffle=True, num_workers=2, collate_fn=lambda x: tuple(zip(*x)))
    
    weights = RetinaNet_ResNet50_FPN_Weights.DEFAULT
    model = retinanet_resnet50_fpn(weights=weights, box_score_thresh=0.95)
    model.eval()
    preprocess = weights.transforms()
    device = "cpu"
    categories = weights.meta["categories"]
    
    pipeline = TorchVisionObjectDetectionPipeline(model=model, 
                                                  preprocessor=preprocess, 
                                                  categories=categories,
                                                  device=device)
    
    show_image = True
    
    for images, _ in data_loader:
        for i in range(len(images)):
            image = images[i]
            #image = T.ToPILImage()()
    
            input_tensor = pipeline.preprocess(image)
            output_tensor = pipeline.predict(input_tensor)
            image_with_boxes = pipeline.postprocess(output_tensor)
            if show_image:
                image_with_boxes.show()
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="Path to the input file")
    parser.add_argument(
        "--model_url",
        type=str,
        default="facebook/detr-resnet-50",
        help="Hugging face model URL. Example 'facebook/detr-resnet-50'",
    )
    args = parser.parse_args()

    main(args.path, args.model_url)