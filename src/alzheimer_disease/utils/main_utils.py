import os
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from pymongo import MongoClient
from bson.binary import Binary
from dotenv import load_dotenv
from pathlib import Path
from alzheimer_disease.logger import logging
from alzheimer_disease.exception import AlzException
import logging
import tensorflow as tf
import sys
import ssl
import random
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import shutil
import sys
import json
from sklearn.metrics import classification_report
import certifi
import base64
load_dotenv()
# Get the MongoDB connection URI from the environment variable

import ssl

# Disable SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

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
        raise AlzException(e,sys)

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
        client = MongoClient(os.getenv("uri"), tlsCAFile=certifi.where())
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






def create_datasets(train_path, test_path, image_size, batch_size, validation_split,test_save_path):
    """
    Create training, validation, and test datasets from the specified directories.

    Args:
        train_path (str): Path to the directory containing the training images.
        test_path (str): Path to the directory containing the test images.
        image_size (tuple): Size of the input images in the format (height, width).
        batch_size (int): Batch size for the datasets.
        validation_split (float): Fraction of the training data to be used for validation.

    Returns:
        train_ds (tf.data.Dataset): Training dataset.
        val_ds (tf.data.Dataset): Validation dataset.
        test_ds (tf.data.Dataset): Test dataset.
        class_names (list): List of class names in the dataset.
    """
    # Define the resize and rescale transformations
   

    # Create training dataset with validation split
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        train_path,
        seed=12,
        shuffle=True,
        image_size=image_size,
        batch_size=batch_size,
        validation_split=validation_split,
        subset="training"
    )

    # Create validation dataset with validation split
    validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        train_path,
        seed=12,
        shuffle=True,
        image_size=image_size,
        batch_size=batch_size,
        validation_split=validation_split,
        subset="validation"
    )

    # Create test dataset
    test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        test_path,
        seed=12,
        shuffle=True,
        image_size=image_size,
        batch_size=batch_size
    )

    # Get the class names from the training dataset
    class_names = train_dataset.class_names

    resize_and_rescale = tf.keras.Sequential([
        tf.keras.layers.Resizing(image_size[0], image_size[1]),
    tf.keras.layers.Rescaling(1./255)
    ])

      # Preprocess and cache the training dataset
    train_ds = train_dataset.map(lambda x, y: (resize_and_rescale(x), y)).cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
    
    # Preprocess and cache the validation dataset
    val_ds = validation_dataset.map(lambda x, y: (resize_and_rescale(x), y)).cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

# Preprocess and cache the test dataset
    test_ds = test_dataset.map(lambda x, y: (resize_and_rescale(x), y)).cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

    tf.data.Dataset.save(test_ds, test_save_path)

    return train_ds, val_ds, test_ds, class_names



def create_callbacks(checkpoint_filepath, patience):
    """
    Create callbacks for model checkpoint and early stopping.

    Returns:
        callbacks (list): List of callbacks.
    """
    
    
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        frequency='epoch',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
   
    callbacks = [model_checkpoint_callback]
    
    return callbacks





def evaluate_model(model, test_ds, class_names, output_file, expected_score, overfitting_threshold):
    """
    Evaluate the trained model on the test dataset and generate evaluation metrics.

    Args:
        model (tf.keras.Model): Trained model.
        test_ds (tf.data.Dataset): Test dataset.
        class_names (list): List of class names.
        output_file (str): File path for saving the evaluation results.
        expected_score (float): Expected accuracy score.
        overfitting_threshold (float): Threshold for detecting overfitting.

    Returns:
        None
    """
    # Make predictions on the test dataset
    y_true = []
    y_pred = []
    for images, labels in test_ds:
        predictions = model.predict(images)
        predicted_labels = tf.argmax(predictions, axis=1).numpy()
        y_true.extend(labels.numpy())
        y_pred.extend(predicted_labels)

    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    # Save the evaluation results as JSON
    with open(output_file, 'w') as f:
        json.dump(report, f)

    # Print the evaluation report
    print("Evaluation results:")
    print(json.dumps(report, indent=4))

    # Calculate f1 scores
    f1_train_score = report["weighted avg"]["f1-score"]
    f1_test_score = report["macro avg"]["f1-score"]

    # Log train and test scores
    logging.info(f"Train score: {f1_train_score} and test score: {f1_test_score}")

    # Check for overfitting or underfitting or expected score
    logging.info("Checking if the model is underfitting or not")
    if f1_test_score < expected_score:
        raise Exception(f"Model is not good as it is not able to give the expected accuracy: "
                        f"expected score: {expected_score}, model actual score: {f1_test_score}")

    logging.info("Checking if the model is overfitting or not")
    diff = abs(f1_train_score - f1_test_score)

    if diff > overfitting_threshold:
        raise Exception(f"Train and test score difference: {diff} is more than the overfitting threshold: "
                        f"{overfitting_threshold}")



def save_object(file_path: str, obj: object) -> None:
    try:
        logging.info("Entered the save_object method of utils")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if isinstance(obj, tf.keras.Model):
            obj.save(file_path)
     
        logging.info("Exited the save_object method of utils")
    except Exception as e:
        raise AlzException(e, sys)
    


def load_object(file_path: str) -> object:
    try:
        logging.info("Entered the load_object method of utils")
        if file_path.endswith(".h5"):
            obj = tf.keras.models.load_model(file_path)
           
        logging.info("Exited the load_object method of utils")
        return obj
    except Exception as e:
        raise AlzException(e, sys)



def decodeImage(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    with open(fileName, 'wb') as f:
        f.write(imgdata)
        f.close()


def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())