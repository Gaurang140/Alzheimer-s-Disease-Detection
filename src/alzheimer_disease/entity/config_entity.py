import os
from dataclasses import dataclass
from datetime import datetime
from alzheimer_disease.constant.training_pipeline import *






TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

@dataclass
class TrainingPipelineConfig:
    artifacts_dir: str = os.path.join(ARTIFACTS_DIR,TIMESTAMP)

training_pipeline_config:TrainingPipelineConfig = TrainingPipelineConfig() 




@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(
        training_pipeline_config.artifacts_dir, DATA_INGESTION_DIR_NAME
    )

    feature_store_file_path: str = os.path.join(
        data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR
    )

    database_name : str = DATA_BASE_NAME
    train_collection_name : str = TRAIN_COLLECTION_NAME
    test_collection_name : str = TEST_COLLECTION_NAME




@dataclass
class DataValidationConfig:
    data_validation_dir: str = os.path.join(
        training_pipeline_config.artifacts_dir, DATA_VALIDATION_DIR_NAME
    )

    valid_status_file_dir: str = os.path.join(data_validation_dir, DATA_VALIDATION_STATUS_FILE)

    required_file_list = DATA_VALIDATION_ALL_REQUIRED_FILES

    


@dataclass
class ModelTrainerConfig:
    model_trainer_dir: str = os.path.join(
        training_pipeline_config.artifacts_dir, MODEL_TRAINER_DIR_NAME)

    model_save: str = os.path.join(model_trainer_dir, MODEL_NAME)
    batch_size : int = MODEL_TRAINER_BATCH_SIZE
    epochs : int = MODEL_TRAINER_EPOCHS
    checkpoint_dir: str = os.path.join(model_trainer_dir, MODEL_CHECKPOINT_DIR_NAME)
    image_size: int = IMAGE_SIZE
    channels: int = CHANNELS
    input_shape: tuple = INPUT_SHAPE
    classes: int = CLASSES
    validation_split: int = VALIDATION_SPLIT
    optimizer: str = OPTIMIZER
    patience: int = PATIENCE
    test_save_dir: str = os.path.join(model_trainer_dir, MODEL_TEST)
    expected_score_threshold: float = EXPECTED_SCORE_THRESHOLD
    overfitting_threshold: float = OVERFITTING_THRESHOLD
    prediction_results_report_dir: str = os.path.join(model_trainer_dir, PREDICTION_REPORT)


@dataclass
class ModelEvaluationConfig:
        change_threshold :float = CHANGE_THRESHOLD





@dataclass
class ModelPusherConfig:
        model_pusher_dir = os.path.join(training_pipeline_config.artifacts_dir , "model_pusher")
        saved_model_dir = os.path.join("saved_models")
        pusher_model_dir = os.path.join(model_pusher_dir,"saved_models")
        pusher_model_path = os.path.join(pusher_model_dir , MODEL_NAME)
        
