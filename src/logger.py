import logging
import os
from datetime import datetime

# Define log file name with current timestamp
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Define the logs directory path relative to the current working directory
logs_path = os.path.join(os.getcwd(), "logs")

# Create the logs directory if it does not exist
os.makedirs(logs_path, exist_ok=True)

# Define the complete log file path
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Configure the basic logging setup
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

if __name__ == "__main__":
    logging.info("Logging has started.")