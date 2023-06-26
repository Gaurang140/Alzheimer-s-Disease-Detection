import numpy as np
import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.preprocessing import image
from alzheimer_disease.config import ModelManager
import os
from alzheimer_disease.logger import logging

PREDICTION_DIR = "predictions"


class PredictPipeline:
    def __init__(self,filename):
        self.filename =filename


    def predict(self):
        os.makedirs(PREDICTION_DIR, exist_ok=True)
        logging.info(f"Creating model resolver object")
        model_resolver = ModelManager(model_registry="saved_models")
        # Load the model
        model = load_model(model_resolver.get_latest_model_path())

        # Define the class names
        class_names = ['Final AD JPEG', 'Final CN JPEG', 'Final EMCI JPEG', 'Final LMCI JPEG', 'Final MCI JPEG']

        # Load and preprocess the test image
        test_image = image.load_img(self.filename, target_size=(256, 256))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image = test_image / 255.0

        # Make predictions
        predictions = model.predict(test_image)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = class_names[predicted_class_index]
        confidence = round(100 * np.max(predictions[0]), 2)

        return [{"image_class": predicted_class, "confidence": confidence}]
