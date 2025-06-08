from flask import Blueprint, request
from flaskr.middlewares.auth import access_token_required
from flaskr.recommender.retrieval import Retrieval
from flaskr.recommender.similar import Similar
from flaskr.recommender.similar import Book
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


@recommends_BP.get("/similar/book/<book>")
@access_token_required
def get_similars(book: str):
    return Similar.get_instance().find_similar_books(book_id=book, top_n=10)


@recommends_BP.post("/similar/book/init")
@access_token_required
def init_similar():
    books = request.get_json()
    if isinstance(books, list):
        if len(books) == 0:
            raise BadRequestError("きっとまた会えるさ")
        books = [Book(**book) for book in books]
        Similar.get_instance().init(books)
        return {"message": "OK"}
    raise BadRequestError("きっとまた会えるさ")


@recommends_BP.post("/similar/book")
@access_token_required
def add_book_similar():
    book = request.get_json()
    book = Book(**book)
    Similar.get_instance().add_book(book)

    return {"message": "OK"}


@recommends_BP.put("/similar/book/<book>")
@access_token_required
def update_book_similar(book: str):
    book_data = request.get_json()
    updated_book = Book(_id=book, **book_data)
    Similar.get_instance().update_book(updated_book)

    return {"message": "OK"}
