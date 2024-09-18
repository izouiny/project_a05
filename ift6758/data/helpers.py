from .api import (ApiClient, FileSystemCache, GameType)
import os

# GLOBAL VARIABLES
cache_path = os.path.dirname(os.path.abspath(__file__)) + "/storage/cache"
cache = FileSystemCache(cache_path)
api_client = ApiClient(cache)
all_seasons = range(2016, 2024)
game_types = [GameType.REGULAR, GameType.PLAYOFF]

def get_all_seasons_games_data() -> list[object]:
    output = list()
    for season in all_seasons:
        data = api_client.get_games_data(season, game_types)
        output.append(data)
    return output

def clear_cache() -> None:
    cache.clear()
