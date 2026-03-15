import logging
import os
from datetime import datetime

# Generate a log file name using the current date and time
# Format: month_day_year_hour_minute_second.log
LOG_FILE = f"{datetime.now().strftime('%m_%d_Y_%H_%M_S')}.log"

# Define the path where logs will be stored (inside a "logs" folder in the current working directory)
log_path = os.path.join(os.getcwd(), "logs", LOG_FILE)

# Create the directory if it does not already exist
os.makedirs(log_path, exist_ok=True)

# Full path to the log file
LOG_FILE_PATH = os.path.join(log_path, LOG_FILE)

# Configure logging settings
logging.basicConfig(
    filename=LOG_FILE_PATH,  # Save logs to the generated file path
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",  # Log format
    level=logging.INFO,  # Set logging level to INFO (only info and above will be logged)
)