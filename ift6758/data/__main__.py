"""
This file is for development and testing purpose
"""
from helpers import get_game_id
from enums import GameType
from classes import (ApiClient, FileSystemCache)
import os

cache_path = os.path.dirname(os.path.abspath(__file__)) + "/storage/cache"
cache = FileSystemCache(cache_path)
client = ApiClient(cache)

game_id = get_game_id(2020, GameType.REGULAR, 23)
data = client.get_game_play_by_play_details(game_id)
