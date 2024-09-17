"""
This file is for development and testing purpose
"""
from helpers import get_game_id
from enums import GameType
from ApiClient import ApiClient

client = ApiClient(True)

print(get_game_id(2020, GameType.REGULAR, 23))
