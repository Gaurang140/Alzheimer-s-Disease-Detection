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

DATA_BASE_NAME = "alzeimer_dataset"
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
CLASSES:int = 4
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




