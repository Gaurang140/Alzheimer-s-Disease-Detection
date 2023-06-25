from alzheimer_disease.utils.main_utils import get_data_from_mongodb 
import os


database_name = "alzeimer_database-1"
data_folder_path =os.path.join("data")
collection_name = "collection_name"

get_data_from_mongodb(database_name=database_name,collection_name=collection_name , output_folder_path=data_folder_path)
