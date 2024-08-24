class ObjectDetectionPipeline:
    def __init__(self, model, preprocessor, postprocessor):
        self.model = model
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

    def predict(self, image):
        preprocessed_image = self.preprocessor.preprocess(image)
        predictions = self.model.predict(preprocessed_image)
        postprocessed_predictions = self.postprocessor.postprocess(predictions)
        return postprocessed_predictions