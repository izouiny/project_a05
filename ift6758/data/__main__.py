"""
This file is for development and testing purpose
"""
from enums import GameType
from classes import (get_game_id, ApiClient, FileSystemCache)
import os

cache_path = os.path.dirname(os.path.abspath(__file__)) + "/storage/cache"
cache = FileSystemCache(cache_path)
client = ApiClient(cache)

game_id = get_game_id(2020, GameType.REGULAR, 23)
data = client.get_game_play_by_play_details(game_id)
