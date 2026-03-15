import sys

def error_message_detail(error, error_details: sys):
    """
    Extracts detailed error information including file name, line number, and error message.

    Parameters
    ----------
    error : Exception
        The actual exception object raised.
    error_details : sys
        The sys module, used to access exception info.

    Returns
    -------
    str
        A formatted string containing file name, line number, and error message.
    """
    # Get traceback object from the exception info
    _, _, exc_tb = error_details.exc_info()
    
    # Extract the file name where the error occurred
    file_name = exc_tb.tb_frame.f_code.co_filename
    
    # Format the error message with file name, line number, and error details
    error_message = "Error Occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )

    return error_message


class ClassifierException(Exception):
    """
    Custom exception class that provides detailed error messages
    including file name and line number.
    """
    def __init__(self, error_message, error_details: sys):
        # Initialize the base Exception class
        super().__init__(error_message)

        # Generate a detailed error message using helper function
        self.error_message = error_message_detail(error_message, error_details)

    def __str__(self):
        # Return the detailed error message when the exception is printed
        return self.error_message