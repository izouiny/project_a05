"""
This file is for development and testing purpose
"""
from ApiClient import ApiClient
from GameType import GameType

client = ApiClient(True)

print(client.get_game_id(2020, GameType.REGULAR, 23))
