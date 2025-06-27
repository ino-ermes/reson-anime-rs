from flaskr.recommender.retrieval_model import RetrievalModel
from typing import List, TypedDict
import tensorflow as tf
import keras
import json
from threading import Lock, Thread
from http import HTTPStatus
import os


class Rating(TypedDict):
    user: str
    book: str
    rating: int


class Retrieval:
    __instance = None

    @staticmethod
    def get_instance():
        if Retrieval.__instance is None:
            Retrieval()
        return Retrieval.__instance

    def __init__(self):
        if Retrieval.__instance is not None:
            raise Exception("This class is a singleton!")

        class ModelHotSwap(TypedDict):
            model: RetrievalModel
            index_to_book: dict[int, str]
            user_to_index: dict[str, int]

        self.__model_hot_swap: ModelHotSwap = None

        self.__train_lock = Lock()
        self.__tranning = False

        Retrieval.__instance = self

    def load(self):
        if not os.path.isfile("data/index_to_book.json"):
            print(f"No data file found at data/index_to_book.json. Starting fresh.")
            return
        if not os.path.isfile("data/user_to_index.json"):
            print(f"No data file found at data/user_to_index.json. Starting fresh.")
            return
        if not os.path.isfile("data/model.keras"):
            print(f"No data file found at data/model.keras. Starting fresh.")
            return

        with open("data/index_to_book.json", "r") as f:
            index_to_book = json.load(f)
            index_to_book = {int(k): v for k, v in index_to_book.items()}

        with open("data/user_to_index.json", "r") as f:
            user_to_index = json.load(f)
            user_to_index = {k: int(v) for k, v in user_to_index.items()}

        model = keras.models.load_model("data/model.keras")

        self.__model_hot_swap = {
            "index_to_book": index_to_book,
            "user_to_index": user_to_index,
            "model": model,
        }

    def __retrain_task(self, ratings: List[Rating]):
        users = sorted({r["user"] for r in ratings})
        books = sorted({r["book"] for r in ratings})

        user_to_index = {u: i for i, u in enumerate(users)}
        book_to_index = {b: i for i, b in enumerate(books)}
        index_to_book = {i: b for b, i in book_to_index.items()}

        users_count = len(user_to_index)
        books_count = len(book_to_index)

        def preprocess_custom_rating(entry):
            user_idx = user_to_index[entry["user"]]
            book_idx = book_to_index[entry["book"]]
            rating = (entry["rating"] - 1.0) / 4.0  # normalize to [0,1]
            return user_idx, {"book_idx": book_idx, "rating": rating}

        processed_ratings = list(map(preprocess_custom_rating, ratings))

        dataset = (
            tf.data.Dataset.from_generator(
                lambda: iter(processed_ratings),
                output_signature=(
                    tf.TensorSpec(shape=(), dtype=tf.int32),
                    {
                        "book_idx": tf.TensorSpec(shape=(), dtype=tf.int32),
                        "rating": tf.TensorSpec(shape=(), dtype=tf.float32),
                    },
                ),
            )
            .shuffle(len(ratings), seed=42, reshuffle_each_iteration=False)
            .batch(1)
            .cache()
        )

        model = RetrievalModel(users_count, books_count)
        model.compile(optimizer=keras.optimizers.Adagrad(learning_rate=0.1))
        model.fit(dataset, epochs=10)

        model.save("data/model.keras")
        with open("data/index_to_book.json", "w") as f:
            json.dump(index_to_book, f)
        with open("data/user_to_index.json", "w") as f:
            json.dump(user_to_index, f)

        self.__model_hot_swap = {
            "index_to_book": index_to_book,
            "user_to_index": user_to_index,
            "model": model,
        }

        with self.__train_lock:
            self.__tranning = False

    def retrain(self, ratings: List[Rating]):
        with self.__train_lock:
            if self.__tranning:
                return {"status": "already retraining"}, HTTPStatus.NOT_MODIFIED
            self.__tranning = True

        Thread(target=self.__retrain_task, args=(ratings,)).start()
        return {"status": "retraining started"}, HTTPStatus.ACCEPTED

    def get_recommends(self, user: str) -> List[str]:
        model_hot_swap = self.__model_hot_swap

        if model_hot_swap is None:
            return []

        user_idx = model_hot_swap["user_to_index"].get(user)
        if user_idx is None:
            return []

        predictions = model_hot_swap["model"](
            keras.ops.convert_to_tensor([user_idx]), training=False
        )
        predictions = keras.ops.convert_to_numpy(predictions["predictions"])

        return [
            model_hot_swap["index_to_book"].get(book_idx) for book_idx in predictions[0]
        ]
