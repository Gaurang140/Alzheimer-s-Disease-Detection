import os
from pymongo import MongoClient
from bson.binary import Binary
from pathlib import Path
from alzheimer_disease.logger import logging
from alzheimer_disease.exception import AlzException

def dump_data_to_mongodb(uri, database_name, data_folder_path, collection_name):
    """
    Dump image data from the specified data folder to MongoDB collection.

    Args:
        uri (str): MongoDB connection URI.
        database_name (str): Name of the database to connect to.
        data_folder_path (str or Path): Path to the data folder containing train and test folders.
        collection_name (str): Name of the collection to store the image data.

    Returns:
        None
    """
    try:
        # Connect to MongoDB server
        client = MongoClient(uri)
        db = client.get_database(database_name)

        # Iterate over train and test folders
        for folder in ["train", "test"]:
            folder_path = data_folder_path / folder

            # Iterate over class folders
            for class_folder in os.listdir(folder_path):
                class_folder_path = folder_path / class_folder

                # Iterate over image files in class folder
                for image_file in os.listdir(class_folder_path):
                    if image_file.endswith(".jpg") or image_file.endswith(".jpeg"):
                        image_file_path = class_folder_path / image_file

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

        logging.info("Data dumped successfully to MongoDB.")
    except Exception as e:
        raise AlzException(e)
    
def get_data_from_mongodb(uri, database_name, collection_name, output_folder_path):
    """
    Retrieve image data from the specified MongoDB collection and save as JPEG files.

    Args:
        uri (str): MongoDB connection URI.
        database_name (str): Name of the database to connect to.
        collection_name (str): Name of the collection to retrieve image data from.
        output_folder_path (str or Path): Path to the output folder to save the JPEG files.

    Returns:
        None
    """
    try:
        # Connect to MongoDB server
        client = MongoClient(uri)
        db = client.get_database(database_name)

        # Convert output_folder_path to Path object if it's a string
        output_folder_path = Path(output_folder_path)

        # Retrieve image data from the collection
        image_collection = db.get_collection(collection_name)
        documents = image_collection.find()

        # Iterate over the retrieved documents
        for document in documents:
            class_label = document["class_label"]
            encoded_image = document["image"]

            # Create output folder for the class label if it doesn't exist
            class_folder_path = output_folder_path / class_label
            os.makedirs(class_folder_path, exist_ok=True)

            # Save the encoded image as JPEG file
            image_file_path = class_folder_path / f"{document['_id']}.jpg"
            with open(image_file_path, "wb") as file:
                file.write(encoded_image)

        logging.info("Data retrieved successfully from MongoDB and saved as JPEG files.")
    except Exception as e:
        raise AlzException(e)