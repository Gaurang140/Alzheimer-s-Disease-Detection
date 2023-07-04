from alzheimer_disease.entity.artifacts_entity import(DataIngestionArtifact,
                                                      DataValidationArtifact,
                                                      ModelTrainerArtifcats,
                                                      ModelEvaluationArtifact,
                                                      ModelPusherArtifact)
                                                    
from alzheimer_disease.entity.config_entity import (DataIngestionConfig,
                                                    TrainingPipelineConfig,
                                                    DataValidationConfig,
                                                    ModelTrainerConfig,
                                                    ModelEvaluationConfig,
                                                    ModelPusherConfig)
from alzheimer_disease.components.data_ingestion import DataIngestion
from alzheimer_disease.components.data_validation import DataValidation
from alzheimer_disease.components.model_trainer import ModelTrainer
from alzheimer_disease.components.model_eval import ModelEvaluation
from alzheimer_disease.components.model_pusher import ModelPusher
from alzheimer_disease.exception import AlzException
import sys


"""
# Retrieve the command-line arguments
activation = str(sys.argv[1])
optimizer = str(sys.argv[2])
batch_size = int(sys.argv[3])
dropout_rate = float(sys.argv[4])
epochs = int(sys.argv[5])
use_early_stopping = sys.argv[6]
load_weights = sys.argv[7]
use_lr_scheduler = sys.argv[8]
"""
activation = "relu"
optimizer = "adam"
batch_size = 32
dropout_rate = 0.1
epochs = 2
use_early_stopping = False
load_weights = False
use_lr_scheduler = False




try :
    #data ingestion
    training_pipeline_config = TrainingPipelineConfig()
    data_ingestion_config  = DataIngestionConfig()

    # Usage example

    data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
    data_ingestion_artifact = data_ingestion.initiate_data_ingestion()




    ########data validation###########


    data_validation_config = DataValidationConfig()
    data_validation = DataValidation(data_validation_config=data_validation_config, data_ingestion_artifact=data_ingestion_artifact)

    data_validation_artifact = data_validation.initialte_data_validation()

    if data_validation_artifact.validation_status ==True:

        ########model trainer###########

            model_trainer_config = ModelTrainerConfig()
            model_trainer = ModelTrainer(model_trainer_config=model_trainer_config, data_ingestion_artifact=data_ingestion_artifact)
            model_trainer_artifact =model_trainer.initiate_model_trainer(
                                                        activation=activation,
                                                        optimizer=optimizer,
                                                        dropout_rate=dropout_rate,
                                                        epochs=epochs,
                                                        batch_size=batch_size,
                                                        use_early_stopping=use_early_stopping,
                                                        load_weights=load_weights,
                                                        use_lr_scheduler=use_lr_scheduler,
                                                        )




            model_eval_config = ModelEvaluationConfig()
            model_eval = ModelEvaluation(model_eval_config=model_eval_config, 
                                        model_trainer_artifact=model_trainer_artifact,
                                        data_ingestion_artifact=data_ingestion_artifact)

            model_eval_artifact = model_eval.initiate_model_evaluation()

        




            ####################-------model pusher-----####################################


            model_pusher_config = ModelPusherConfig()
            model_pusher = ModelPusher(model_pusher_config=model_pusher_config, model_trainer_artifact=model_trainer_artifact)
            model_pusher_artifact = model_pusher.initiate_model_pusher()
        
    else : 
        raise AlzException("Data Validation Failed")
    
except Exception as e:
    raise AlzException(str(e),sys)
        

        
  
  