from alzheimer_disease.pipeline.s01_data_ingestion import DataIngestionArtifact 
from alzheimer_disease.utils.main_utils import create_callbacks, create_datasets, evaluate_model
from alzheimer_disease.exception import AlzException
from alzheimer_disease.logger import logging
from alzheimer_disease.entity.artifacts_entity import ModelTrainerArtifcats
from alzheimer_disease.entity.config_entity import ModelTrainerConfig
from keras.applications import EfficientNetB0, VGG19
from mlflow import log_metric, log_param, log_artifacts
from keras.callbacks import EarlyStopping, LearningRateScheduler
import mlflow
import traceback 
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
    
    def build_model(self, input_shape, activation='relu', dropout_rate=0.3):
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
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Flatten(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(64, activation=activation),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(256, activation=activation),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(128, activation=activation),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(5, activation='softmax')
        ])

        return model
    
    def train_model(self, train_ds=None, test_ds=None, val_ds=None, activation=None, 
                    dropout_rate=None, epochs=None, batch_size=None, use_early_stopping=None,
                    load_weights=None, use_lr_scheduler=None):
        try:
            model = self.build_model(input_shape=(self.model_trainer_config.image_size[0],
                                  self.model_trainer_config.image_size[1],
                                  self.model_trainer_config.channels)
                                  , activation=activation, dropout_rate=dropout_rate)

            model.compile(optimizer=self.model_trainer_config.optimizer,
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                          metrics=['accuracy'])

            callbacks = []
            callbacks += create_callbacks(self.model_trainer_config.checkpoint_dir, patience=self.model_trainer_config.patience)

            if use_early_stopping:
                early_stop = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
                callbacks.append(early_stop)

            if use_lr_scheduler:
                def lr_scheduler(epoch, lr):
                    if epoch < 50:
                        return lr
                    else:
                        return lr * tf.math.exp(-0.1)

                learning_rate_scheduler = LearningRateScheduler(lr_scheduler)
                callbacks.append(learning_rate_scheduler)

            #if load_weights:
                #model.load_weights(self.model_trainer_config.best_model_weights)

            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks
            )

            for metric_name, metric_value in history.history.items():
                mlflow.log_metric(metric_name, metric_value[-1])

            logging.info("Training Completed")

            return model
        except Exception as e:
            raise AlzException(traceback.format_exc())
    
    def initiate_model_trainer(self, activation=None, optimizer=None, dropout_rate=None,
                               epochs=None, batch_size=None, use_early_stopping=None,
                               load_weights=None, use_lr_scheduler=None):
        logging.info("Preparing TensorFlow Dataset for training")
        train, val, test, class_name = create_datasets(train_path=self.data_ingestion_artifact.train_path,
                                                      test_path=self.data_ingestion_artifact.test_path,
                                                      image_size=self.model_trainer_config.image_size,
                                                      batch_size=self.model_trainer_config.batch_size,
                                                      validation_split=self.model_trainer_config.validation_split,
                                                      test_save_path=self.model_trainer_config.test_save_dir)
            
        logging.info(f"TensorFlow Dataset prepared and has {class_name}")



          # Log the parameters for MLflow
        mlflow.log_param("activation", activation)
        mlflow.log_param("optimizer", optimizer)
        mlflow.log_param("dropout_rate", dropout_rate)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("use_early_stopping", use_early_stopping)
        mlflow.log_param("load_weights", load_weights)
        mlflow.log_param("use_lr_scheduler", use_lr_scheduler)







        logging.info("Train the model")
        model = self.train_model(
            train_ds=train,
            test_ds=test,
            val_ds=val,
            activation=activation,
            dropout_rate=dropout_rate,
            epochs=epochs,
            batch_size=batch_size,
            use_early_stopping=use_early_stopping,
            load_weights=load_weights,
            use_lr_scheduler=use_lr_scheduler)
        
        evaluate_model(model, test, class_name,
                       self.model_trainer_config.prediction_results_report_dir,
                       self.model_trainer_config.expected_score_threshold,
                       self.model_trainer_config.overfitting_threshold)
        
        model.save(self.model_trainer_config.model_save)
        
        logging.info("Model Saved")
        
        model_trainer_artifact = ModelTrainerArtifcats(model_dir=self.model_trainer_config.model_save,
                                                       test_path=self.model_trainer_config.test_save_dir,
                                                       eval_report=self.model_trainer_config.prediction_results_report_dir)
        
        logging.info(f"Model Trained Artifact Created: {model_trainer_artifact}")
        
        return model_trainer_artifact
