from alzheimer_disease.entity.artifacts_entity import(DataIngestionArtifact)                            
from alzheimer_disease.entity.config_entity import (DataIngestionConfig)
from alzheimer_disease.components.data_ingestion import DataIngestion
from alzheimer_disease.exception import AlzException
from alzheimer_disease.logger import logging
import sys


STAGE_NAME = "Data Ingestion stage"

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        data_ingestion_config = DataIngestionConfig()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.initiate_data_ingestion()



if __name__ == '__main__':
    try:
        logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logging.exception(e)
        raise e