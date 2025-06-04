from http import HTTPStatus

def default_handler(e):
    return {
        "message": str(e),
        "status coe": HTTPStatus.INTERNAL_SERVER_ERROR,
    }, HTTPStatus.INTERNAL_SERVER_ERROR
