from src.classifier import logging, ClassifierException
import sys

if __name__ == "__main__":
    logging.info("The application has started ")
    try:
        # code 
        a = 1/1
        logging.info("execution successfully")
    except Exception as e:
        logging.info("Custom Exception at app.py file")
        raise ClassifierException(e , sys)