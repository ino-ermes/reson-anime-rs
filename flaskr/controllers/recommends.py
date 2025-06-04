from flask import Blueprint, request
from flaskr.middlewares.auth import access_token_required
from flaskr.recommender.retrieval import Retrieval
from flaskr.errors.bad_request import BadRequestError

recommends_BP = Blueprint("recommends", __name__, url_prefix="/api/v1/recommends")


@recommends_BP.get("/retrieval/user/<user>")
@access_token_required
def get_recommend(user: str):
    return Retrieval.get_instance().get_recommends(user)


@recommends_BP.post("/retrieval/user")
@access_token_required
def retrain():
    ratings = request.get_json()
    if isinstance(ratings, list):
        if len(ratings) == 0:
            raise BadRequestError("きっとまた会えるさ")
        return Retrieval.get_instance().retrain(ratings)
    raise BadRequestError("きっとまた会えるさ")
