from flask import Flask
from werkzeug.exceptions import HTTPException
from dotenv import load_dotenv


def create_app():
    app = Flask(__name__, instance_relative_config=True)

    load_dotenv()

    # json converter
    from flaskr.utils.json_helper import CustomJSONProvider

    app.json = CustomJSONProvider(app)

    # error handler
    from flaskr.errors.bad_request import BadRequestError
    from flaskr.errors.not_found import NotFoundError
    from flaskr.errors.unauthenicated import UnauthenticatedError
    from flaskr.errors.forbidden import ForbiddenError
    from flaskr.errorHandlers.my_handler import my_handler
    from flaskr.errorHandlers.default_http_handler import default_http_handler
    from flaskr.errorHandlers.default_handler import default_handler

    app.register_error_handler(BadRequestError, my_handler)
    app.register_error_handler(NotFoundError, my_handler)
    app.register_error_handler(UnauthenticatedError, my_handler)
    app.register_error_handler(ForbiddenError, my_handler)
    app.register_error_handler(HTTPException, default_http_handler)
    app.register_error_handler(Exception, default_handler)

    # api/v1/recommend
    from flaskr.controllers.recommends import recommends_BP

    app.register_blueprint(recommends_BP)

    # from flaskr.recommender.recommender import Recommender

    # Recommender.get_instance().load()

    return app
