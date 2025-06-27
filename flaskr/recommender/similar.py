import numpy as np
import pickle
import os
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import threading


class Book:
    """Represents a book with its basic attributes."""

    def __init__(self, _id: str, title: str, description: str):
        self._id = _id
        self.title = title
        self.description = description


class Similar:
    """
    A thread-safe singleton class to manage book embeddings and find similarities,
    with persistence to save and load data.
    """

    __instance = None
    _lock = threading.Lock()  # Class-level lock for thread-safe operations

    @staticmethod
    def get_instance(file_path="data/mbeddings.pkl"):
        """Static access method for the singleton instance."""
        # Use a lock to ensure thread-safe singleton creation
        with Similar._lock:
            if Similar.__instance is None:
                Similar(file_path)
        return Similar.__instance

    def __init__(self, file_path="data/embeddings.pkl"):
        """
        Virtually private constructor.
        Initializes the model, embeddings, and mapping dictionaries.
        Attempts to load data from the specified file path upon creation.
        """
        if Similar.__instance is not None:
            raise Exception("This class is a singleton! Use get_instance().")

        self.embeddings: np.ndarray = np.array([])
        # Uses a pre-trained model for creating sentence/text embeddings.
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.id_to_index: Dict[str, int] = {}
        self.index_to_id: Dict[int, str] = {}
        self.file_path = file_path

        # Attempt to load existing data
        self.load_data()

        Similar.__instance = self

    def save_data(self):
        """Saves embeddings and mappings to a file using pickle."""
        try:
            with open(self.file_path, "wb") as f:
                data = {
                    "embeddings": self.embeddings,
                    "id_to_index": self.id_to_index,
                    "index_to_id": self.index_to_id,
                }
                pickle.dump(data, f)
            print(f"Data successfully saved to {self.file_path}")
        except IOError as e:
            print(f"Error saving data to {self.file_path}: {e}")

    def load_data(self):
        """Loads embeddings and mappings from a file if it exists."""
        if not os.path.exists(self.file_path):
            print(f"No data file found at {self.file_path}. Starting fresh.")
            return

        try:
            with open(self.file_path, "rb") as f:
                data = pickle.load(f)
                self.embeddings = data["embeddings"]
                self.id_to_index = data["id_to_index"]
                self.index_to_id = data["index_to_id"]
            print(f"Data successfully loaded from {self.file_path}")
        except (IOError, pickle.UnpicklingError, KeyError) as e:
            print(f"Error loading data from {self.file_path}: {e}. Starting fresh.")
            # Clear potentially corrupt data
            self.embeddings = np.array([])
            self.id_to_index = {}
            self.index_to_id = {}

    def init(self, books: List[Book]) -> None:
        """
        Compute embeddings for an initial list of books.
        This will overwrite any existing embeddings.

        Args:
            books: A list of Book objects.
        """
        with self._lock:  # Ensure thread-safe modification of shared data
            # Concatenate title and description to form the text for embedding.
            texts = [f"{book.title} {book.description}" for book in books]
            embeddings = self.model.encode(texts, convert_to_numpy=True)

            self.embeddings = embeddings
            self.id_to_index = {book._id: i for i, book in enumerate(books)}
            self.index_to_id = {i: book._id for i, book in enumerate(books)}
            # After re-initializing, save the new state
            self.save_data()

    def add_book(self, new_book: Book) -> None:
        """
        Add a new book and its embedding to the existing set.
        If the book already exists, it will not be added again.

        Args:
            new_book: The Book object to add.
        """
        with self._lock:  # Ensure thread-safe modification
            # Prevent adding a book with a duplicate ID.
            if self.id_to_index.get(new_book._id) is not None:
                print(f"Book with ID {new_book._id} already exists.")
                return

            # Determine the new index for the book.
            new_index = (max(self.index_to_id.keys()) + 1) if self.index_to_id else 0
            self.id_to_index[new_book._id] = new_index
            self.index_to_id[new_index] = new_book._id

            # Compute embedding for the new book.
            new_text = f"{new_book.title} {new_book.description}"
            new_embedding = self.model.encode([new_text], convert_to_numpy=True)

            # Append the new embedding to the existing embeddings array.
            if self.embeddings.size == 0:
                self.embeddings = new_embedding
            else:
                self.embeddings = np.vstack([self.embeddings, new_embedding])

            # Save after adding
            self.save_data()

    def update_book(self, updated_book: Book) -> None:
        """
        Update an existing book's details and recompute its embedding.

        Args:
            updated_book: The Book object with updated information.
        """
        with self._lock:  # Ensure thread-safe modification
            book_index = self.id_to_index.get(updated_book._id)
            if book_index is None:
                print(f"Book with ID {updated_book._id} not found for update.")
                return

            # Recompute the embedding with the new text and update it in place.
            new_text = f"{updated_book.title} {updated_book.description}"
            new_embedding = self.model.encode([new_text], convert_to_numpy=True)
            self.embeddings[book_index] = new_embedding[0]
            # Save after updating
            self.save_data()

    def find_similar_books(self, book_id: str, top_n: int = 12) -> List[str]:
        """
        Find the most similar books to a given book ID. This is a read operation
        and can be concurrent, but locking provides consistency if data is being modified.
        """
        with self._lock:  # Lock to ensure we get a consistent read of the data
            if self.embeddings.size == 0:
                return []

            book_index = self.id_to_index.get(book_id)
            if book_index is None:
                print(f"Book with ID {book_id} not found.")
                return []

            # Ensure the index is within the bounds of the embeddings array
            if book_index >= len(self.embeddings):
                print(
                    f"Error: Index for book ID {book_id} is out of bounds for embeddings."
                )
                return []

            target_embedding = self.embeddings[book_index]
            similarities = cosine_similarity(
                target_embedding.reshape(1, -1), self.embeddings
            )[0]
            similar_indices = sorted(
                enumerate(similarities), key=lambda x: x[1], reverse=True
            )

            similar_book_ids = [
                self.index_to_id[index]
                for index, score in similar_indices[1 : top_n + 1]
            ]

            return similar_book_ids

    def recommend_for_user(
        self, liked_book_ids: List[str], top_n: int = 12
    ) -> List[str]:
        """
        Recommend books for a user based on their liked books using content-based filtering.

        Args:
            liked_book_ids: List of book IDs the user liked/read.
            top_n: Number of recommendations to return.

        Returns:
            A list of recommended book IDs, excluding already liked ones.
        """
        with self._lock:
            if not liked_book_ids:
                print("No liked books provided for recommendation.")
                return []

            valid_indices = [
                self.id_to_index.get(book_id)
                for book_id in liked_book_ids
                if book_id in self.id_to_index
            ]

            if not valid_indices:
                print("No valid liked book IDs found in index.")
                return []

            # Get embeddings for the liked books
            liked_embeddings = self.embeddings[valid_indices]

            # Compute the user profile vector (mean of liked embeddings)
            user_profile = np.mean(liked_embeddings, axis=0)

            # Compute similarity with all books
            similarities = cosine_similarity(
                user_profile.reshape(1, -1), self.embeddings
            )[0]

            # Rank all books by similarity
            ranked_indices = sorted(
                enumerate(similarities), key=lambda x: x[1], reverse=True
            )

            # Exclude books the user has already liked
            liked_index_set = set(valid_indices)
            recommended_book_ids = []
            for index, _ in ranked_indices:
                if index not in liked_index_set:
                    recommended_book_ids.append(self.index_to_id[index])
                if len(recommended_book_ids) >= top_n:
                    break

            return recommended_book_ids
