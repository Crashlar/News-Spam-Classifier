import os
from pathlib import Path
import logging

# Configure logging to display informational messages
logging.basicConfig(level=logging.INFO)

# Define the project name
project_name = "classifier"


# List of files and directories to be created for the project structure
list_of_files = [
    f"src/{project_name}/__init__.py",

    # Components folder and files
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_transformation.py",
    f"src/{project_name}/components/model_training.py",
    f"src/{project_name}/components/model_monitoring.py",

    # Pipelines folder and files
    f"src/{project_name}/pipelines/__init__.py",
    f"src/{project_name}/pipelines/training_pipeline.py",
    f"src/{project_name}/pipelines/prediction_pipeline.py",

    # Utility and core files
    f"src/{project_name}/exception.py",
    f"src/{project_name}/logger.py",
    f"src/{project_name}/utils.py",

    # Root-level files
    "app.py",
    "Dockerfile",
    ".dockerignore",   
    "requirements.txt",
    "setup.py",
    "notebooks/experiment1.ipynb",
    ".env",
    
    # kaggle secrets files
    "~/.kaggle/.gitignore",
    "~/.kaggle/kaggle.json",
    
]



# Iterate through each file path in the list
for filepath in list_of_files:
    filepath = Path(filepath)  # Convert to Path object for consistency
    filedir, filename = os.path.split(filepath)  # Separate directory and filename

    # If the file has a directory, create it if it doesn't exist
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file {filename}")

    # Create the file if it does not exist or is empty
    if (not os.path.exists(filepath) or (os.path.getsize(filepath) == 0)):
        with open(filepath, "w") as f:
            pass  # Just create an empty file
            logging.info(f"Creating empty file: {filename}")
    else:
        # If the file already exists and is not empty, log that information
        logging.info(f"{filename} already exists")