import os
import tensorflow as tf
import mlflow
from mlflow import log_metric, log_param, log_artifact




ARTIFACTS_DIR: str = "artifacts"


"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_DIR_NAME: str = "data_ingestion"

DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"

DATA_BASE_NAME = "alzeimer_database-1"
OUTPUT_FOLDER_PATH =os.path.join("data")
#COLLECTION_NAME = "collection_name"
#VAL_RATION = 0.1
#TRAIN_RATIO = 0.8
TRAIN_COLLECTION_NAME = "train"
TEST_COLLECTION_NAME = "test"





"""
Data Validation realted contant start with DATA_VALIDATION VAR NAME
"""

DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_STATUS_FILE = 'status.txt'
DATA_VALIDATION_ALL_REQUIRED_FILES = ["train", "test"]




"""
Data Transformation related constant start with MODEL_TRAINER var name
"""
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_FEATURE_STORE_DIR: str = "feature_store"
MODEL_NAME :str = "alzeimer_model.h5"
MODEL_TRAINER_BATCH_SIZE : int = 32
MODEL_TRAINER_EPOCHS : int = 1
MODEL_CHECKPOINT_DIR_NAME: str =  "checkpoints"
IMAGE_SIZE: tuple = (35, 35)
CHANNELS: int = 3
INPUT_SHAPE: tuple = (IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
CLASSES:int = 5
VALIDATION_SPLIT: int = 0.2
OPTIMIZER: str = "adam"
PATIENCE :int = 10
MODEL_TEST = "Tensorflow_test"
EXPECTED_SCORE_THRESHOLD : float = 0.1
PREDICTION_REPORT:str = 'evaluation_results.json'
OVERFITTING_THRESHOLD :float = 0.30






"""
MODEL PUSHER related constant start with MODEL_PUSHER var name
"""
CHANGE_THRESHOLD: float = 0.05





# Log parameters using mlflow.log_param()
mlflow.log_param("ARTIFACTS_DIR", ARTIFACTS_DIR)
mlflow.log_param("DATA_INGESTION_DIR_NAME", DATA_INGESTION_DIR_NAME)
mlflow.log_param("DATA_INGESTION_FEATURE_STORE_DIR", DATA_INGESTION_FEATURE_STORE_DIR)
mlflow.log_param("DATA_BASE_NAME", DATA_BASE_NAME)
mlflow.log_param("OUTPUT_FOLDER_PATH", OUTPUT_FOLDER_PATH)
mlflow.log_param("TRAIN_COLLECTION_NAME", TRAIN_COLLECTION_NAME)
mlflow.log_param("TEST_COLLECTION_NAME", TEST_COLLECTION_NAME)
mlflow.log_param("DATA_VALIDATION_DIR_NAME", DATA_VALIDATION_DIR_NAME)
mlflow.log_param("DATA_VALIDATION_STATUS_FILE", DATA_VALIDATION_STATUS_FILE)
mlflow.log_param("DATA_VALIDATION_ALL_REQUIRED_FILES", str(DATA_VALIDATION_ALL_REQUIRED_FILES))
mlflow.log_param("MODEL_TRAINER_DIR_NAME", MODEL_TRAINER_DIR_NAME)
mlflow.log_param("MODEL_TRAINER_FEATURE_STORE_DIR", MODEL_TRAINER_FEATURE_STORE_DIR)
mlflow.log_param("MODEL_NAME", MODEL_NAME)
mlflow.log_param("MODEL_TRAINER_BATCH_SIZE", str(MODEL_TRAINER_BATCH_SIZE))
mlflow.log_param("MODEL_TRAINER_EPOCHS", str(MODEL_TRAINER_EPOCHS))
mlflow.log_param("MODEL_CHECKPOINT_DIR_NAME", MODEL_CHECKPOINT_DIR_NAME)
mlflow.log_param("IMAGE_SIZE", str(IMAGE_SIZE))
mlflow.log_param("CHANNELS", str(CHANNELS))
mlflow.log_param("INPUT_SHAPE", str(INPUT_SHAPE))
mlflow.log_param("CLASSES", str(CLASSES))
mlflow.log_param("VALIDATION_SPLIT", str(VALIDATION_SPLIT))
mlflow.log_param("OPTIMIZER", OPTIMIZER)
mlflow.log_param("PATIENCE", str(PATIENCE))
mlflow.log_param("MODEL_TEST", MODEL_TEST)
mlflow.log_param("EXPECTED_SCORE_THRESHOLD", str(EXPECTED_SCORE_THRESHOLD))
mlflow.log_param("PREDICTION_REPORT", PREDICTION_REPORT)
mlflow.log_param("OVERFITTING_THRESHOLD", str(OVERFITTING_THRESHOLD))
mlflow.log_param("CHANGE_THRESHOLD", str(CHANGE_THRESHOLD))