from http import HTTPStatus


def default_http_handler(e):
    message = e.description or "Some thing is wrong"
    status_code = (
        e.get_response().status_code
        if e.get_response()
        else HTTPStatus.INTERNAL_SERVER_ERROR
    )

    return {"message": message, "status coe": status_code}, status_code
