import requests
import json

from .Cache import Cache

class ApiClient:

    """
    This class allows to interact with the API of the NHL
    """

    base_url = "https://api-web.nhle.com/v1"

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
