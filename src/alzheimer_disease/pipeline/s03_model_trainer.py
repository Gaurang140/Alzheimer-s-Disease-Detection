from alzheimer_disease.entity.artifacts_entity import DataIngestionArtifact, ModelTrainerArtifcats
from alzheimer_disease.entity.config_entity import ModelTrainerConfig
from alzheimer_disease.components.model_trainer import ModelTrainer
from alzheimer_disease.exception import AlzException
from alzheimer_disease.logger import logging
import sys

STAGE_NAME = "Model Training stage"

class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        model_trainer_config = ModelTrainerConfig()
        data_ingestion_artifact =DataIngestionArtifact(train_path=r'artifacts\data_ingestion\train', 
                                                        test_path=r'artifacts\data_ingestion\test')  

        model_trainer = ModelTrainer(
            model_trainer_config=model_trainer_config,
            data_ingestion_artifact=data_ingestion_artifact
        )
        model_trainer.initiate_model_training()




if __name__ == '__main__':
    try:
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainerTrainingPipeline()
        obj.main()
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logging.exception(e)
        raise e
