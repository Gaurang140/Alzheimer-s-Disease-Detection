import os
from pymongo import MongoClient
from bson.binary import Binary
from dotenv import load_dotenv
from pathlib import Path
from src.alzheimer_disease.utils.main_utils import dump_data_to_mongodb
load_dotenv()

# Load environment variables from .env file
load_dotenv()

# Get the MongoDB connection URI from the environment variable
uri = os.getenv("uri")
database_name = "alzeimer_database-1"
data_folder_path = Path("D:\Internship_Project\Alzeimer_Diasis_Project\ALZ_Project_Code\Data\Alzheimers-ADNI")
collection_name = "collection_name"

dump_data_to_mongodb(uri=uri , database_name=database_name, data_folder_path=data_folder_path, collection_name=collection_name)


