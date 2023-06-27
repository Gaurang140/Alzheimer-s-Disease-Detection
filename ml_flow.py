from alzheimer_disease.entity.artifacts_entity import (DataIngestionArtifact, DataValidationArtifact,
                                                        ModelTrainerArtifcats, ModelEvaluationArtifact,
                                                        ModelPusherArtifact)
from alzheimer_disease.entity.config_entity import (DataIngestionConfig, TrainingPipelineConfig,
                                                    DataValidationConfig, ModelTrainerConfig,
                                                    ModelEvaluationConfig, ModelPusherConfig)
from alzheimer_disease.components.data_ingestion import DataIngestion
from alzheimer_disease.components.data_validation import DataValidation
from alzheimer_disease.components.model_trainer import ModelTrainer
from alzheimer_disease.components.model_eval import ModelEvaluation
from alzheimer_disease.components.model_pusher import ModelPusher
from alzheimer_disease.exception import AlzException
from alzheimer_disease.constant import *
import sys
import mlflow
import os
from urllib.parse import urlparse
from mlflow import log_metric, log_param, log_artifacts
import mlflow.tensorflow 


try:
    
    if mlflow.active_run():
        mlflow.end_run()
    # Start MLflow run
    mlflow.start_run()






    # Data ingestion
    training_pipeline_config = TrainingPipelineConfig()
    data_ingestion_config = DataIngestionConfig()

    # Usage example
    data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
    data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

    # Log data ingestion artifact as artifact
    



    


    # Data validation
    data_validation_config = DataValidationConfig()
    data_validation = DataValidation(data_validation_config=data_validation_config,
                                     data_ingestion_artifact=data_ingestion_artifact)
    data_validation_artifact = data_validation.initialte_data_validation()

    # Log data validation artifact as artifact
    mlflow.log_artifact(data_validation_config.data_validation_dir, "data_validation_artifact")

    # Track data validation artifact with DVC
   

    if data_validation_artifact.validation_status:
        # Model training
        model_trainer_config = ModelTrainerConfig()
        model_trainer = ModelTrainer(model_trainer_config=model_trainer_config,
                                     data_ingestion_artifact=data_ingestion_artifact)
        model_trainer_artifact = model_trainer.initiate_model_training()

        # Log model trainer artifact as artifact
        mlflow.log_artifact(model_trainer_artifact.model_dir, "model_trainer_artifact")
        mlflow.log_artifact(model_trainer_artifact.eval_report, "model_trainer_artifact")



         #Track model trainer artifact with DVC
     

        # Model evaluation
        model_eval_config = ModelEvaluationConfig()
        model_eval = ModelEvaluation(model_eval_config=model_eval_config,
                                     model_trainer_artifact=model_trainer_artifact,
                                     data_ingestion_artifact=data_ingestion_artifact)
        model_eval_artifact = model_eval.initiate_model_evaluation()

   
        # Model pushing
        model_pusher_config = ModelPusherConfig()
        model_pusher = ModelPusher(model_pusher_config=model_pusher_config,
                                   model_trainer_artifact=model_trainer_artifact)
        model_pusher_artifact = model_pusher.initiate_model_pusher()

        # Log model pusher artifact as artifact
        mlflow.log_artifact(model_pusher_artifact.pusher_model_dir, "model_pusher_artifact")
        mlflow.log_artifact(model_pusher_artifact.saved_model_dir, "model_pusher_artifact")

       





        MLFLOW_TRACKING_URI= "https://dagshub.com/Gaurang140/Alzheimer-s-Disease-Detection.mlflow"
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

   

      
    else:
        raise AlzException("Data Validation Failed")

except Exception as e:
    raise AlzException(str(e), sys)

finally:
    # End MLflow run
    mlflow.end_run()

