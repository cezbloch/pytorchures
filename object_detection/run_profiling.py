import argparse

from object_detection.pipeline import ObjectDetectionPipeline


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

    pipeline = ObjectDetectionPipeline(args.model_url)
    pipeline.run()
