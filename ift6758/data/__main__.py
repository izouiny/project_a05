"""
This file is for development and testing purpose
"""
from helpers import get_game_id
from enums import GameType
from classes import (ApiClient, FileSystemCache)
import os

client = ApiClient(True)

print(get_game_id(2020, GameType.REGULAR, 23))

cache_path = os.path.dirname(os.path.abspath(__file__)) + "/storage/cache"
cache = FileSystemCache(cache_path)

cache.set("test/123", "{}")
print(cache.get("test/123"))
print(cache.has("test/123"))
print(cache.remove("test/123"))

print(cache.get("test/456"))
print(cache.has("test/456"))
print(cache.remove("test/456"))

cache.set("test/7889", "{}")
print(cache.clear())