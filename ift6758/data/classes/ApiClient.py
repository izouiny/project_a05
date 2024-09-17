from typing import List

import requests
import json

from .Cache import Cache

def get_game_id(season: int, game_type: str, game_number: int) -> str:
    """
    Compute a game id

    Args:
        season (int): the season for which to retrieve the game ID (use starting year of the season)
        game_type (int): type of game
        game_number (int): identify the specific game number.

    Returns:
        list: The game id as string
    """
    return str(season) + game_type + ('%04d' % game_number)


class ApiClient:

    """
    This class allows to interact with the API of the NHL
    """

    base_url = "https://api-web.nhle.com/v1"

    max_game_number = 1800

    def __init__(self, cache: Cache = None):
        """
        Args:
            cache: Cache engine to use for caching API responses. This limits call done to the API.
        """
        self.cache = cache


    def get_game_play_by_play_details(self, game_id: str) -> object:
        """
        Get data from API or from cache is cache storage is provided
        """

        # Define url path. It will be used for cache key as well
        uri = "/gamecenter/" + game_id + "/play-by-play"

        # Try to get result from cache
        if self.cache is not None:
            value = self.cache.get(uri)
            if value is not None:
                print("Load " + uri + " from cache")
                return json.loads(value)

        # Get content from API or raise an error
        full_url = self.base_url + uri
        response = requests.get(full_url)
        response.raise_for_status()
        text = response.text

        # Store the content if cache is enabled
        if self.cache is not None:
            self.cache.set(uri, text)

        return json.loads(text)

    def get_game_data(self, season: int, game_types: list[str]) -> list[object] :
        """
        Get data from an entire season
        """

