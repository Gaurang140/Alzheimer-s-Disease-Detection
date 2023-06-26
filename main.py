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
            model_trainer_artifact = model_trainer.initiate_model_training()




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
        

        
  
  