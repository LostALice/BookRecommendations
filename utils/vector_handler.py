# Code by AkinoAlice@TyrantRey

from text2vec import SentenceModel
from pydantic import BaseModel
from tqdm import tqdm

import numpy as np


class BookContext(BaseModel):
    title: str
    context: list[str]


class VectorHandler(object):
    def __init__(
        self, embedding_model_name: str = "shibing624/text2vec-base-chinese"
    ) -> None:
        self.HF_embedding_model = embedding_model_name
        assert self.HF_embedding_model, "No specific embedding model"
        self.embedding = SentenceModel(self.HF_embedding_model)

    def encoder(self, text: str | list[str]) -> np.ndarray:
        """convert text to ndarray (vector)

        Args:
            text (str): text to be converted

        Returns:
            ndarray: numpy array (vector)
        """
        return np.array(self.embedding.encode(text))

    def text_split(self, title: str = "", context: str = "") -> BookContext:
        book_context = [text for text in context.split("\n\n")]

        return BookContext(title=title, context=book_context)

    def save_to_file(self, vector: list[np.ndarray], filename: str = "unnamed") -> str:
        """save vector to file

        Args:
            vector (list[np.ndarray]): list of numpy arrays
            filename (str, optional): save filename. Defaults to "unnamed".

        Returns:
            str: save file path
        """
        _save_path = f"./vector/{filename}.npz"
        np.savez(file=_save_path, vector=vector)
        return _save_path

    def read_from_file(self, vector_path: str) -> tuple[str, list[np.ndarray]]:
        """load vector from file

        Args:
            vector_path (str): path to vector

        Returns:
            list[np.ndarray]: return list of numpy arrays
        """
        file_name = vector_path.split("/")[-1]
        _vector = np.load(vector_path)["vector"]

        return file_name, _vector

    def calculate_similarity(self, query_vector: np.ndarray, books_vectors: list[np.ndarray]) -> float:
        """return the minimum similarity between vectors

        Args:
            query_vector (np.ndarray): ocr vector result
            books_vectors (list[np.ndarray]): load from file vector

        Returns:
            float: distance between query vector and books vectors
        """
        distances = [np.linalg.norm(query_vector - book_vector)
                     for book_vector in books_vectors]
        return np.min(distances)


# running testing on ./
if __name__ == "__main__":
    from pprint import pprint

    book_name = "Twelve Years a Slave"

    with open(f"./books/{book_name}.txt", "r", encoding="utf-8") as book:
        txt = book.read()

    vector_Handler = VectorHandler()
    book_context = vector_Handler.text_split(
        title=book_name, context=txt)

    vector = []
    for i in tqdm(book_context.context):
        vector.append(vector_Handler.encoder(i))
    vector_Handler.save_to_file(vector, book_name)
