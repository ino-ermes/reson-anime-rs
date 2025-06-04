from flaskr.errors.custom_api import CustomAPIError
from http import HTTPStatus

class NotFoundError(CustomAPIError):

    def __init__(self, message):
        super().__init__(message)
        self.status_code = HTTPStatus.NOT_FOUND