from alzheimer_disease.entity.artifacts_entity import DataValidationArtifact,DataIngestionArtifact
from alzheimer_disease.entity.config_entity import DataValidationConfig
from alzheimer_disease.components.data_validation import DataValidation
from alzheimer_disease.exception import AlzException
from alzheimer_disease.logger import logging

import sys

STAGE_NAME = "Data Validation stage"

class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        data_validation_config = DataValidationConfig()
        data_ingestion_artifact = DataIngestionArtifact(train_path='artifacts\data_ingestion\train ', 
                                                        test_path='artifacts\data_ingestion\test')  # Replace with the actual DataIngestionArtifact object

        data_validation = DataValidation(
            data_validation_config=data_validation_config,
            data_ingestion_artifact=data_ingestion_artifact
        )
        data_validation_artifact = data_validation.initialte_data_validation()

        if data_validation_artifact.validation_status:
            logging.info(f">>>>>> stage {STAGE_NAME} completed successfully <<<<<<")
        else:
            raise AlzException("Data Validation Failed")


if __name__ == '__main__':
    try:
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataValidationTrainingPipeline()
        obj.main()
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logging.exception(e)
        raise e
