from alzheimer_disease.entity.artifacts_entity import ModelPusherArtifact,ModelTrainerArtifcats
from alzheimer_disease.entity.config_entity import ModelPusherConfig
from alzheimer_disease.components.model_pusher import ModelPusher
from alzheimer_disease.exception import AlzException
from alzheimer_disease.logger import logging
import sys

STAGE_NAME = "Model Pusher stage"

class ModelPusherTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        model_pusher_config = ModelPusherConfig()
        model_trainer_artifact = ModelTrainerArtifcats()  

        model_pusher = ModelPusher(
            model_pusher_config=model_pusher_config,
            model_trainer_artifact=model_trainer_artifact
        )
        model_pusher.initiate_model_pusher()

        

if __name__ == '__main__':
    try:
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelPusherTrainingPipeline()
        obj.main()
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logging.exception(e)
        raise e
