from setuptools import setup , find_packages
import os 
from typing import List

EMAIL = os.getenv("AUTHOR_EMAIL")
AUTHOR_NAME = os.getenv("AUTHOR_NAME")

HYPEN_E_DOT = "-e ."


def get_requirement_packages(file_path :str)-> List[str]:
    """
    Reads a requirements file and returns a list of dependencies.

    Parameters
    ----------
    file_path : str
        The path to the requirements file (e.g., 'requirements.txt').

    Returns
    -------
    List[str]
        A list of requirement strings, with newline characters removed.
    
    Notes
    -----
    - Each line in the file is treated as one requirement.
    - Newline characters at the end of each line are stripped.
    """
    
    requirements = []
    
    # Open the file in read mode
    with open(file_path) as file_obj:
        # Read all lines from the file
        requirements = file_obj.readlines()
        
        # Remove newline characters from each line
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements



setup(
    # --Basic info--
    name="News-Spam-Classifier",
    version="0.0.1",
    author = AUTHOR_NAME, 
    author_email= EMAIL,
    description="Build a Machine Learning model that can automatically detect fake or spam news articles",
    url="https://github.com/Crashlar/News-Spam-Classifier",
    
    # package configuration 
    packages=find_packages(),
    python_requires=">=3.11",
    
    
    # dependencies
    install_requires=get_requirement_packages("requirements.txt")
    
    
    

)

