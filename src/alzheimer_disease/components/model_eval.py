import logging
import tensorflow as tf
from sklearn.metrics import f1_score
from alzheimer_disease.entity import artifacts_entity 
from alzheimer_disease.entity import config_entity
from alzheimer_disease.exception import AlzException
from alzheimer_disease.logger import logging 
from alzheimer_disease.config import ModelManager
import sys 
import mlflow 
import shutil


class ModelEvaluation:
    def __init__(self, model_eval_config: config_entity.ModelEvaluationConfig,
                 data_ingestion_artifact: artifacts_entity.DataIngestionArtifact,
                 model_trainer_artifact: artifacts_entity.ModelTrainerArtifcats):
        try:
            logging.info(f"{'>>'*20}  Model Evaluation {'<<'*20}")
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.model_resolver = ModelManager()
        except Exception as e:
            raise AlzException(e, sys)

    def initiate_model_evaluation(self) -> artifacts_entity.ModelEvaluationArtifact:
        try:
            logging.info("If the saved model folder has a model, we will compare "
                         "which model is better trained: the saved model or the current model")

            latest_dir_path = self.model_resolver.get_latest_dir_path()
            if latest_dir_path is None:
                model_eval_artifact = artifacts_entity.ModelEvaluationArtifact(is_model_accepted=True,
                                                                               improved_accuracy=None)
                logging.info(f"Model evaluation artifact: {model_eval_artifact}")
                return model_eval_artifact

            # Load the saved model
            model_path = self.model_resolver.get_latest_model_path()
            logging.info("Loading previously trained model")
            previous_model = tf.keras.models.load_model(model_path)

            # Load the current model
            logging.info("Loading currently trained model")
            current_model = tf.keras.models.load_model(self.model_trainer_artifact.model_dir)

            # Evaluate models on the test dataset
            test = self.model_trainer_artifact.test_path
            test_ds = tf.data.Dataset.load(test)
            y_true = []
            y_pred_previous = []
            y_pred_current = []
            for data in test_ds:
                images = data[0]  # Extract images from the data
                labels = data[1]  # Extract labels from the data
                predictions_previous = previous_model.predict(images)
                predicted_labels_previous = tf.argmax(predictions_previous, axis=1).numpy()
                y_true.extend(labels.numpy())
                y_pred_previous.extend(predicted_labels_previous)

                predictions_current = current_model.predict(images)
                predicted_labels_current = tf.argmax(predictions_current, axis=1).numpy()
                y_pred_current.extend(predicted_labels_current)

              


            # Calculate F1 scores
            previous_model_score = f1_score(y_true=y_true, y_pred=y_pred_previous, average='macro')
            current_model_score = f1_score(y_true=y_true, y_pred=y_pred_current,average='macro')

            logging.info(f"Accuracy using the previously trained model: {previous_model_score}")
            logging.info(f"Accuracy using the currently trained model: {current_model_score}")

            mlflow.log_metric("previous_model_score", previous_model_score)
            mlflow.log_metric("current_model_score", current_model_score)

            if current_model_score <= previous_model_score:
                logging.info("The currently trained model is not better than the previous model")
                raise Exception("The currently trained model is not better than the previous model")

            model_eval_artifact = artifacts_entity.ModelEvaluationArtifact(is_model_accepted=True,
                                                                           improved_accuracy=current_model_score - previous_model_score)
            logging.info(f"Model evaluation artifact: {model_eval_artifact}")

            return model_eval_artifact
        except Exception as e:
            raise AlzException(e, sys)
