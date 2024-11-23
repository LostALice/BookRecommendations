# Code by AkinoAlice@TyrantRey


import numpy as np
import sqlite3


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
