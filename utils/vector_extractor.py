# Code by AkinoAlice@TyrantRey

from text2vec import SentenceModel

import numpy as np


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


if __name__ == "__main__":
    vector_extractor = VectorExtractor()
    print(vector_extractor.encoder("how are you?").shape)
