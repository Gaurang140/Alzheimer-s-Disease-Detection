import os
from pymongo import MongoClient
from bson.binary import Binary
from pathlib import Path
from src.alzheimer_disease.utils.main_utils import dump_data_to_mongodb
import ssl

# Disable SSL verification
ssl._create_default_https_context = ssl._create_unverified_context
# Get the MongoDB connection URI from the environment variable
database_name = "alzeimer_dataset"
data_folder_path = Path(r"C:\Users\Gaurang\Downloads\archive\Alzheimer_s Dataset")
train_collection_name = "train"
test_collection_name = "test"

dump_data_to_mongodb(database_name=database_name, data_folder_path=data_folder_path, train_collection_name=train_collection_name,
                    test_collection_name=test_collection_name)


