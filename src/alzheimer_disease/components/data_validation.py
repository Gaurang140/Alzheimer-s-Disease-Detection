from alzheimer_disease.entity.artifacts_entity import DataIngestionArtifact,DataValidationArtifact
from alzheimer_disease.entity.config_entity import DataValidationConfig
from alzheimer_disease.exception import AlzException
from alzheimer_disease.logger import logging 
import sys 
import shutil
import os  # Add this line to import the os module
from mlflow import log_metric, log_param, log_artifacts


class DataValidation:
    def __init__(self, data_ingestion_artifact:DataIngestionArtifact,
                 data_validation_config:DataValidationConfig):
        logging.info(f"{'>>'*20} Data Validation{'<<'*20}")
        
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
        except Exception as e:
            raise AlzException(e,sys)
        
    import os

    def validate_data(self) -> bool:
        try:
            train_folder_path = self.data_ingestion_artifact.train_path
            test_folder_path = self.data_ingestion_artifact.test_path

            train_folder_exists = os.path.exists(train_folder_path)
            test_folder_exists = os.path.exists(test_folder_path)

            validation_status = train_folder_exists and test_folder_exists

            os.makedirs(self.data_validation_config.data_validation_dir, exist_ok=True)
            with open(self.data_validation_config.valid_status_file_dir, 'w') as f:
                f.write(f"validation_status: {validation_status}\n")

            return validation_status
        except Exception as e:
            raise AlzException(e, sys)

        

    def initialte_data_validation (self)->DataValidationArtifact:
        logging.info("Entered initiate_data_validation method of DataValidation class")

        try :

            status = self.validate_data()

            data_validation_artifact = DataValidationArtifact(validation_status=status)

            logging.info("Exited data_validation method of DataValidation class")
            logging.info("data_validation_artifact: {}".format(data_validation_artifact))

    
            return data_validation_artifact
        except Exception as e:
            raise AlzException(e,sys)
