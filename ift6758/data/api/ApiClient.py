import requests
import json

from .enums import GameType
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

    games_base_url = "https://api-web.nhle.com/v1"

    stats_base_url = "https://api.nhle.com/stats/rest/en"

    def __init__(self, cache: Cache = None):
        """
        Args:
            cache: Cache engine to use for caching API responses. This limits call done to the API.
        """
        self.cache = cache

    def get_games_data(self, season: int, game_types: list[str]) -> list[object] :
        """
        Get data from an entire season.
        This methods will request the API a lot of time.
        """
        games = []

        for game_type in game_types:
            # Retrieve game count for this season and type
            game_count = self.get_game_count_in_season(season, game_type)
            print("[ApiClient.get_games_data] Found %d games for season %d and type %s" % (game_count, season, game_type))

            # If no game for this type game, continue
            if game_count == 0:
                continue

            for game_number in range(1, game_count):
                # Get game data from game_id
                game_id = get_game_id(season, game_type, game_number)
                game_data = self.get_game_data(game_id)
                games.append(game_data)

        return games

    def get_game_data(self, game_id: str) -> object:
        """
        Get data from API or from cache is cache storage is provided
        """

        # Define url path. It will be used for cache key as well
        uri = "/gamecenter/" + game_id + "/play-by-play"
        cache_key = "games/" + uri

        # Try to get result from cache
        if self.cache is not None:
            value = self.cache.get(cache_key)
            if value is not None:
                print("[ApiClient.get_game_data] Loaded '%s' from cache" % uri)
                return json.loads(value)

        # Get content from API or raise an error
        full_url = self.games_base_url + uri
        response = requests.get(full_url)
        response.raise_for_status()
        text = response.text

        print("[ApiClient.get_game_data] Loaded '%s' from API" % uri)

        # Store the content if cache is enabled
        if self.cache is not None:
            self.cache.set(cache_key, text)

        return json.loads(text)


    def get_game_count_in_season(self, season: int, game_type: str) -> int:
        """
        Get the number of games in a season
        Raises an error if the game_type is not recognized
        Args:
            season: First year of the season to retrieve, i.e. for the 2016-2017 season you'd put in 2016
            game_type: type of the game
        """
        season_data = self.get_season_data(season)

        if season_data is None:
            print("[ApiClient.get_game_count_in_season] Season '%d' not found" % season)
            return 0

        if game_type == GameType.REGULAR:
            return int(season_data["totalRegularSeasonGames"])
        elif game_type == GameType.PLAYOFF:
            return int(season_data["totalPlayoffGames"])
        else:
            raise Exception("Game type '%s' not recognized" % game_type)


    def get_season_data(self, season: int) -> dict[str | int] | None:
        """
        Get the season data from a specific season
        Args:
            season: First year of the season to retrieve, i.e. for the 2016-2017 season you'd put in 2016
        """
        all_seasons_data = self.get_seasons_data()

        season_id = str(season) + str(season + 1)

        for season_data in all_seasons_data:
            if str(season_data["id"]) == season_id:
                return season_data

        return None


    def get_seasons_data(self) -> list[dict[str | int]]:
        """
        Get data from all seasons.
        Keep results in cache
        """

        # Define url path. It will be used for cache key as well
        uri = "/season"
        cache_key = "stats/" + uri

        # Try to get result from cache
        if self.cache is not None:
            value = self.cache.get(cache_key)
            if value is not None:
                print("[ApiClient.get_seasons_data] Loaded '%s' from cache" % uri)
                return json.loads(value)["data"]

        # Get content from API or raise an error
        full_url = self.stats_base_url + uri
        response = requests.get(full_url)
        response.raise_for_status()
        text = response.text

        print("[ApiClient.get_seasons_data] Loaded '%s' from API" % uri)

        # Store the content if cache is enabled
        if self.cache is not None:
            self.cache.set(cache_key, text)

        return json.loads(text)["data"]