from alzheimer_disease.entity.artifacts_entity import DataIngestionArtifact
from alzheimer_disease.entity.config_entity import DataIngestionConfig,TrainingPipelineConfig
from alzheimer_disease.components.data_ingestion import DataIngestion
from alzheimer_disease.exception import AlzException
import sys




try :
    #data ingestion
    training_pipeline_config = TrainingPipelineConfig()
    data_ingestion_config  = DataIngestionConfig()

    # Usage example

    data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
    data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

except Exception as e:
    raise AlzException(e,sys)