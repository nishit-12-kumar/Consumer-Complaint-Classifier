import os
import sys
import pickle
import yaml
from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    """
    Saves a python object (like tokenizer or label encoder) as a pickle file.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
            
        logging.info(f"Successfully saved object at {file_path}")
        
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    """
    Loads a python object from a pickle file.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def read_yaml_config(file_path):
    """
    Reads the config.yaml file and returns a dictionary.
    """
    try:
        with open(file_path, "r") as file_obj:
            config = yaml.safe_load(file_obj)
            logging.info(f"Successfully read configuration from {file_path}")
            return config
    except Exception as e:
        raise CustomException(e, sys)