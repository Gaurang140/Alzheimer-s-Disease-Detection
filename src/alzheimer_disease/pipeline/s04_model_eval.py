from alzheimer_disease.entity.artifacts_entity import ModelEvaluationArtifact,ModelTrainerArtifcats,DataIngestionArtifact
from alzheimer_disease.entity.config_entity import ModelEvaluationConfig
from alzheimer_disease.components.model_eval import ModelEvaluation
from alzheimer_disease.exception import AlzException
from alzheimer_disease.logger import logging
import sys

STAGE_NAME = "Model Evaluation stage"

class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        model_eval_config = ModelEvaluationConfig()
        model_trainer_artifact = ModelTrainerArtifcats()  # Replace with the actual ModelTrainerArtifact object
        data_ingestion_artifact = DataIngestionArtifact()  # Replace with the actual DataIngestionArtifact object

        model_eval = ModelEvaluation(
            model_eval_config=model_eval_config,
            model_trainer_artifact=model_trainer_artifact,
            data_ingestion_artifact=data_ingestion_artifact
        )
        model_eval_artifact = model_eval.initiate_model_evaluation()

        if model_eval_artifact.evaluation_status:
            logging.info(f">>>>>> stage {STAGE_NAME} completed successfully <<<<<<")
        else:
            raise AlzException("Model Evaluation Failed")


if __name__ == '__main__':
    try:
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelEvaluationTrainingPipeline()
        obj.main()
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logging.exception(e)
        raise e
