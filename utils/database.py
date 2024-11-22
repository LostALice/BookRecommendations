# Code by AkinoAlice@TyrantRey


from pymilvus import MilvusClient
from pymilvus import DataType
from typing import Literal

import numpy as np
import sqlite3


class SetupVectorDatabase(object):
    def __init__(
        self,
        host: str = "localhost",
        port: int = 3306,
        default_collection_name: str = "default",
        debug: bool = False
    ) -> None:

        self.HOST = host
        self.PORT = port
        self.debug = debug
        self.default_collection_name = default_collection_name

        assert self.HOST, "No host specified"
        assert self.PORT, "No port specified"
        assert self.default_collection_name, "No collection specified"

        self.milvus_client = MilvusClient(
            uri=f"http://{self.HOST}:{self.PORT}")

        assert self.milvus_client, "No server available"

        try:
            if self.debug == "True":
                self.milvus_client.drop_collection(
                    collection_name=self.default_collection_name
                )
        finally:
            loading_status = self.milvus_client.get_load_state(
                collection_name=self.default_collection_name
            )

        if (
            not loading_status
            or loading_status["state"] == loading_status["state"].NotExist
        ):
            self.create_collection(
                collection_name=self.default_collection_name)

    def create_collection(
        self,
        collection_name: str,
        index_type: Literal[
            "FLAT",
            "IVF_FLAT",
            "IVF_SQ8",
            "IVF_PQ",
            "HNSW",
            "ANNOY",
            "RHNSW_FLAT",
            "RHNSW_PQ",
            "RHNSW_SQ",
        ] = "IVF_FLAT",
        metric_type: Literal["L2", "IP"] = "L2",
    ) -> dict:

        schema = MilvusClient.create_schema(
            auto_id=True,
            enable_dynamic_field=False,
        )

        schema.add_field(
            field_name="id", datatype=DataType.VARCHAR, max_length=512, is_primary=True
        )
        # file_id
        schema.add_field(
            field_name="source", datatype=DataType.VARCHAR, max_length=1024
        )
        schema.add_field(
            field_name="file_uuid", datatype=DataType.VARCHAR, max_length=36
        )
        schema.add_field(
            field_name="content", datatype=DataType.VARCHAR, max_length=2048
        )
        schema.add_field(field_name="vector",
                         datatype=DataType.FLOAT_VECTOR, dim=768)

        index_params = self.milvus_client.prepare_index_params()

        index_params.add_index(
            field_name="vector",
            index_type=index_type,
            metric_type=metric_type,
            params={"nlist": 128},
        )

        self.milvus_client.create_collection(
            collection_name=collection_name,
            index_params=index_params,
            metric_type=metric_type,
            schema=schema,
        )

        collection_status = self.milvus_client.get_load_state(
            collection_name=collection_name
        )

        return collection_status


class VectorDatabase(SetupVectorDatabase):
    def __init__(
        self,
        host: str = "localhost",
        port: int = 3306,
        default_collection_name: str = "default",
        debug: bool = False
    ) -> None:
        super().__init__(
            host,
            port,
            default_collection_name,
            debug
        )

    def insert_sentence(
        self,
        docs_filename: str,
        vector: np.ndarray,
        content: str,
        book_uuid: str,
        collection: str = "default",
        remove_duplicates: bool = True,
    ) -> dict:
        """insert a sentence(context) from book

        Args:
            docs_filename (str): docs filename
            vector (ndarray): vector of sentences
            content (str): docs content
            book_uuid (str): book_uuid
            collection (str, optional): insert into which collection. Defaults to "default".
            remove_duplicates (bool, optional): remove duplicates vector in database. Defaults to True.

        Returns:
            dict: _description_
        """

        # fix duplicates
        if remove_duplicates:
            is_duplicates = self.milvus_client.query(
                collection_name=collection,
                filter=f"""(source == "{docs_filename}") and (content == "{content}")""",
            )  # nopep8
            if is_duplicates:
                info = self.milvus_client.delete(
                    collection_name="default", ids=[i["id"] for i in is_duplicates]
                )

        success = self.milvus_client.insert(
            collection_name=collection,
            data={
                "source": str(docs_filename),
                "vector": vector,
                "content": content,
                "file_uuid": book_uuid,
            },
        )

        return success


class SetupSQLiteDatabase(object):
    def __init__(
        self,
        db_file: str = "book_recommendations.db",
        debug: bool = False
    ) -> None:
        ...


class SQLiteDatabase(SetupSQLiteDatabase):
    def __init__(
        self,
        db_file: str = "book_recommendations.db",
        debug: bool = False
    ) -> None:
        super().__init__(db_file, debug)


# class RAGHandler(object):
#     def __init__(self) -> None:
#         # only except .gguf format
#         self.model_name = os.getenv("LLM_MODEL")

#         if self.model_name and self.model_name.endswith(".gguf"):
#             raise FormatError

#         if not os.path.exists(f"""./model/{os.getenv("LLM_MODEL")}"""):
#             raise FileNotFoundError

#         # todo: using ollama
#         self.model = Llama(
#             model_path=f"""./model/{os.getenv("LLM_MODEL")}""",
#             verbose=False,
#             n_gpu_layers=-1,
#             n_ctx=0,
#         )

#         self.system_prompt = "你是一個逢甲大學的學生助理，你只需要回答關於學分，課程，老師等有關資料，不需要回答學分，課程，老師以外的問題。你現在有以下資料 {regulations} 根據上文回答問題"

#         self.converter = opencc.OpenCC("s2tw.json")


#     def token_counter(self, prompt: str) -> int:
#         return len(self.model.tokenize(prompt.encode("utf-8")))

#     def response(self, question: str, regulations: list, max_tokens: int = 8192) -> tuple[str, int]:
#         # todo
#         """response from RAG

#         Args:
#             question (str): question from user
#             max_tokens (int, optional): max token allowed. Defaults to 8192.

#         Returns:
#             answer: response from RAG
#             token_size: token size
#         """
#         content = self.system_prompt.format(regulations=" ".join(regulations))

#         token_size = self.token_counter(content)

#         message = [
#             {
#                 "role": "system",
#                 "content": content,
#             },
#             {
#                 "role": "user",
#                 "content": question
#             },
#         ]

#         output = self.model.create_chat_completion(
#             message,
#             stop=["<|eot_id|>", "<|end_of_text|>"],
#             max_tokens=max_tokens,
#             temperature=.5
#         )["choices"][0]["message"]["content"]

#         return str(self.converter.convert(output)), token_size
