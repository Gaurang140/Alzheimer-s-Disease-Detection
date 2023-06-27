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

        model_trainer_artifact = ModelTrainerArtifcats(model_dir="artifacts\model_trainer\alzeimer_model.h5",
                                                       test_path="artifacts\model_trainer\Tensorflow_test",
                                                       eval_report="artifacts\model_trainer\evaluation_results.json"

                                                       )
          # Replace with the actual ModelTrainerArtifact object
        data_ingestion_artifact = DataIngestionArtifact(train_path='artifacts\data_ingestion\train ', 
                                                        test_path='artifacts\data_ingestion\test')   # Replace with the actual DataIngestionArtifact object

        model_eval = ModelEvaluation(
            model_eval_config=model_eval_config,
            model_trainer_artifact=model_trainer_artifact,
            data_ingestion_artifact=data_ingestion_artifact
        )
        model_eval_artifact = model_eval.initiate_model_evaluation()



if __name__ == '__main__':
    try:
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelEvaluationTrainingPipeline()
        obj.main()
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logging.exception(e)
        raise e
