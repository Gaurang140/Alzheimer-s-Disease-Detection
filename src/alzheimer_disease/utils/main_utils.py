import os
from pymongo import MongoClient
from bson.binary import Binary
from dotenv import load_dotenv
from pathlib import Path
from alzheimer_disease.logger import logging
from alzheimer_disease.exception import AlzException
import logging
import random
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import shutil
import sys
load_dotenv()
# Get the MongoDB connection URI from the environment variable


def dump_data_to_mongodb(database_name, data_folder_path, train_collection_name, test_collection_name):
    """
    Dump image data from the specified data folder to MongoDB collections.

    Args:
        database_name (str): Name of the database to connect to.
        data_folder_path (str or Path): Path to the data folder containing train and test folders.
        train_collection_name (str): Name of the collection to store the train image data.
        test_collection_name (str): Name of the collection to store the test image data.

    Returns:
        None
    """
    try:
        # Connect to MongoDB server
        client = MongoClient(os.getenv("uri"))
        db = client.get_database(database_name)

        # Process train folder
        train_folder_path = os.path.join(data_folder_path, "train")
        process_folder(train_folder_path, db, train_collection_name)

        # Process test folder
        test_folder_path = os.path.join(data_folder_path, "test")
        process_folder(test_folder_path, db, test_collection_name)

        logging.info("Data dumped successfully to MongoDB.")
    except Exception as e:
        logging.exception("Failed to dump data to MongoDB.")
        raise AlzException(e)

def process_folder(folder_path, db, collection_name):
    # Iterate over class folders
    for class_folder in os.listdir(folder_path):
        class_folder_path = os.path.join(folder_path, class_folder)

        # Iterate over image files in class folder
        for image_file in os.listdir(class_folder_path):
            if image_file.endswith(".jpg") or image_file.endswith(".jpeg"):
                image_file_path = os.path.join(class_folder_path, image_file)

                # Open and encode the image
                with open(image_file_path, "rb") as file:
                    image_data = file.read()
                    encoded_image = Binary(image_data)

                # Create a document with class label and encoded image
                document = {
                    "class_label": class_folder,
                    "image": encoded_image
                }

                # Store the document in the collection
                image_collection = db.get_collection(collection_name)
                image_collection.insert_one(document)



def get_data_from_mongodb(database_name, train_collection_name, test_collection_name, output_folder_path):
    """
    Retrieve image data from the specified MongoDB collections and save as JPEG files.

    Args:
        database_name (str): Name of the database to connect to.
        train_collection_name (str): Name of the train collection to retrieve image data from.
        test_collection_name (str): Name of the test collection to retrieve image data from.
        output_folder_path (str or Path): Path to the output folder to save the JPEG files.

    Returns:
        (str, str): Paths of the train and test folders.
    """
    try:
        # Connect to MongoDB server
        client = MongoClient(os.getenv("uri"))
        db = client.get_database(database_name)

        # Convert output_folder_path to Path object if it's a string
        output_folder_path = Path(output_folder_path)

        # Create train and test folders
        train_folder_path = output_folder_path / "train"
        test_folder_path = output_folder_path / "test"
        os.makedirs(train_folder_path, exist_ok=True)
        os.makedirs(test_folder_path, exist_ok=True)

        # Retrieve train data from the train collection
        train_collection = db.get_collection(train_collection_name)
        train_documents = train_collection.find()

        # Iterate over the retrieved train documents
        for document in train_documents:
            class_label = document["class_label"]
            encoded_image = document["image"]

            # Create output folder for the class label if it doesn't exist
            class_folder_path = train_folder_path / class_label
            os.makedirs(class_folder_path, exist_ok=True)

            # Save the encoded image as JPEG file
            image_file_path = class_folder_path / f"{document['_id']}.jpg"
            with open(image_file_path, "wb") as file:
                file.write(encoded_image)

        # Retrieve test data from the test collection
        test_collection = db.get_collection(test_collection_name)
        test_documents = test_collection.find()

        # Iterate over the retrieved test documents
        for document in test_documents:
            class_label = document["class_label"]
            encoded_image = document["image"]

            # Create output folder for the class label if it doesn't exist
            class_folder_path = test_folder_path / class_label
            os.makedirs(class_folder_path, exist_ok=True)

            # Save the encoded image as JPEG file
            image_file_path = class_folder_path / f"{document['_id']}.jpg"
            with open(image_file_path, "wb") as file:
                file.write(encoded_image)

        logging.info("Data retrieved successfully from MongoDB and saved as JPEG files.")

        # Return the paths of the train and test folders
        return str(train_folder_path), str(test_folder_path)
    except Exception as e:
        logging.exception("Failed to retrieve data from MongoDB and save as JPEG files.")
        raise AlzException(e,sys)








