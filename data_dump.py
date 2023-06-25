import os
from pymongo import MongoClient
from bson.binary import Binary
from pathlib import Path
from src.alzheimer_disease.utils.main_utils import dump_data_to_mongodb

# Get the MongoDB connection URI from the environment variable
database_name = "alzeimer_database-2"
data_folder_path = Path("D:\Internship_Project\Alzeimer_Diasis_Project\ALZ_Project_Code\Data\Alzheimers-ADNI")
train_collection_name = "train"
test_collection_name = "test"

dump_data_to_mongodb(database_name=database_name, data_folder_path=data_folder_path, train_collection_name=train_collection_name,
                    test_collection_name=test_collection_name)


