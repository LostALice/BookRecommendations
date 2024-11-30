# Code by AkinoAlice@TyrantRey

from text2vec import SentenceModel
from pydantic import BaseModel
from tqdm import tqdm

import numpy as np


class BookContext(BaseModel):
    title: str
    context: list[str]


class VectorExtractor(object):
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


# running testing on ./
if __name__ == "__main__":
    from pprint import pprint

    book_name = "Twelve Years a Slave"

    with open(f"./books/{book_name}.txt", "r", encoding="utf-8") as book:
        txt = book.read()

    vector_extractor = VectorExtractor()
    book_context = vector_extractor.text_split(
        title=book_name, context=txt)

    vector = []
    for i in tqdm(book_context.context):
        vector.append(vector_extractor.encoder(i))
    vector_extractor.save_to_file(vector, book_name)