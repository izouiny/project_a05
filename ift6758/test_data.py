"""
This file is for development and testing purpose
"""
from data import (ApiClient, FileSystemCache, GameType)
import os

cache_path = os.path.dirname(os.path.abspath(__file__)) + "/data/storage/cache"
cache = FileSystemCache(cache_path)
client = ApiClient(cache)

data = client.get_games_data(2020, [GameType.REGULAR, GameType.PLAYOFF])
