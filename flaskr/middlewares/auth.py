from functools import wraps
from flask import request
import os
from flaskr.errors.unauthenicated import UnauthenticatedError


def access_token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if "Authorization" in request.headers:
            try:
                token = request.headers["Authorization"].split(" ")[1]
            except:
                raise UnauthenticatedError("Invalid Authentication Header!")
        if not token:
            raise UnauthenticatedError("Authentication Token is missing!")

        if token != os.getenv("ACCESS_TOKEN"):
            raise UnauthenticatedError("Invalid Authentication token!")

        return f(*args, **kwargs)

    return decorated
