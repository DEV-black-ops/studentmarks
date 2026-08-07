import sys
import src.mlproject.logger as logging


def error_message_detail(error_message, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f"Error occurred in script: [{file_name}] at line number: [{exc_tb.tb_lineno}] error message: [{error_message}]"
    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys = None):
        super().__init__(error_message, error_detail)
        if error_detail is not None:
            self.error_message = error_message_detail(error_message, error_detail)
        else:
            self.error_message = str(error_message)

    def __str__(self):
        return self.error_message