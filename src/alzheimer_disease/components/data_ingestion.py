import os
import sys
from alzheimer_disease.exception import AlzException
from alzheimer_disease.utils.main_utils import get_data_from_mongodb
from alzheimer_disease.logger import logging
from alzheimer_disease.entity.config_entity import DataIngestionConfig
from alzheimer_disease.entity.artifacts_entity import DataIngestionArtifact


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        """
        Initialize the DataIngestion object.

        Args:
            data_ingestion_config (DataIngestionConfig): Configuration object for data ingestion.

        Raises:
            AlzException: If an error occurs during initialization.
        """
        try:
            logging.info(f"{'>>'*20} Data Ingestion {'<<'*20}")
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            logging.exception("Error occurred during DataIngestion initialization.")
            raise AlzException(e)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Perform data ingestion from MongoDB to the feature store folder.

        Returns:
            DataIngestionArtifact: An object containing the paths to the train and test folders.

        Raises:
            AlzException: If an error occurs during data ingestion.
        """
        try:
            logging.info("Collecting data from MongoDB...")
            feature_store_dir = os.path.dirname(self.data_ingestion_config.feature_store_file_path)
            logging.info("Create feature store folder if not available")
            os.makedirs(feature_store_dir, exist_ok=True)

            logging.info("Save data to feature store folder")
            train_path, test_path = get_data_from_mongodb(
                database_name=self.data_ingestion_config.database_name,
                train_collection_name=self.data_ingestion_config.train_collection_name,
                test_collection_name=self.data_ingestion_config.test_collection_name,
                output_folder_path=feature_store_dir
            )

            data_ingestion_artifact = DataIngestionArtifact(train_path=train_path, test_path=test_path)

            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact

        except Exception as e:
            logging.exception("Failed to initiate data ingestion.")
            raise AlzException(e,sys)
