from alzheimer_disease.entity.artifacts_entity import DataIngestionArtifact
from alzheimer_disease.utils.main_utils import create_callbacks, create_datasets,evaluate_model
from alzheimer_disease.exception import AlzException
from alzheimer_disease.logger import logging
from alzheimer_disease.entity.artifacts_entity import DataIngestionArtifact,ModelTrainerArtifcats
from alzheimer_disease.entity.config_entity import ModelTrainerConfig
from keras.applications import EfficientNetB0,VGG19
import ssl

# Disable SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

import tensorflow as tf
from keras import layers
import sys


class ModelTrainer:

    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, model_trainer_config: ModelTrainerConfig) -> ModelTrainerArtifcats:
        logging.info(f"{'>>'*20}Model Training{'<<'*20}")
        self.data_ingestion_artifact = data_ingestion_artifact
        self.model_trainer_config = model_trainer_config
    
    def build_model(self, input_shape):
        """
        Build and compile the convolutional neural network model.

        Args:
            input_shape (tuple): Shape of the input images in the format (height, width, channels).

        Returns:
            model (tf.keras.Model): Compiled convolutional neural network model.
        """
        

      
        base_model = VGG19(include_top=False, weights='imagenet', input_shape=input_shape)
        base_model.trainable = False



        data_augmentation = tf.keras.Sequential([
                    layers.RandomFlip("horizontal_and_vertical"),
                    layers.RandomRotation(0.2),
                ])


        model = tf.keras.models.Sequential([
              
                data_augmentation,
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dense(256, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(5, activation='softmax')
                ])

        return model
        
    def initiate_model_training (self):
        logging.info("Preparing TensorFlow Dataset for training")
        train, val, test, class_name = create_datasets(train_path=self.data_ingestion_artifact.train_path, 
                                                       test_path=self.data_ingestion_artifact.test_path,
                                                       image_size=self.model_trainer_config.image_size, 
                                                       batch_size=self.model_trainer_config.batch_size,
                                                       validation_split=self.model_trainer_config.validation_split,
                                                       test_save_path=self.model_trainer_config.test_save_dir  )
            
        logging.info(f"TensorFlow Dataset prepared and has {class_name}")
        
        model = self.build_model((self.model_trainer_config.image_size[0],
                                  self.model_trainer_config.image_size[1],
                                  self.model_trainer_config.channels))
        
        logging.info("Convolutional Neural Network Model Built")
        
        model.compile(optimizer=self.model_trainer_config.optimizer,
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['accuracy'])
        
        logging.info("Convolutional Neural Network Model Compiled")
        
        callbacks = create_callbacks(self.model_trainer_config.checkpoint_dir, patience=self.model_trainer_config.patience)
        
        logging.info("Callbacks created")
        
        history = model.fit(train,
                            validation_data=val,
                            epochs=self.model_trainer_config.epochs,
                            callbacks=callbacks)
        
        logging.info("Training Completed")

      
        evaluate_model(model,test,class_name,
                       self.model_trainer_config.prediction_results_report_dir,
                       self.model_trainer_config.expected_score_threshold , 
                       self.model_trainer_config.overfitting_threshold)
        
        model.save(self.model_trainer_config.model_save)
        
        logging.info("Model Saved")
        
        model_trainer_artifact = ModelTrainerArtifcats(model_dir=self.model_trainer_config.model_save,
                                                       test_path=self.model_trainer_config.test_save_dir)
        
        logging.info(f"Model Trained Artifact Created: {model_trainer_artifact}")
        
        return model_trainer_artifact
