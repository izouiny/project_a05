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

def get_series_letter(round: int, series: int) -> str:
    """
    Compute the series letter for a given round and series
    """
    round_base_index = 0
    for i in range(1, round):
        round_base_index += 2 ** (4 - i)
    letter_index = round_base_index + series
    series_letter = chr(64 + letter_index)
    return series_letter


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

    def get_games_data(self, season: int, game_types: list[str]) -> list[dict] :
        """
        Get data from an entire season.
        This methods will request the API a lot of time.
        """
        games = []

        for game_type in game_types:
            # Retrieve games number list for this season and type
            game_numbers = self.get_game_numbers_in_season(season, game_type)

            # If no game for this type game, continue
            if game_numbers is None or len(game_numbers) == 0:
                print("[ApiClient.get_games_data] No games for season %d and type %s" % (season, game_type))
                continue

            print("[ApiClient.get_games_data] Found %d games for season %d and type %s" % (len(game_numbers), season, game_type))

            for game_number in game_numbers:
                # Get game data from game_id
                game_id = get_game_id(season, game_type, game_number)
                game_data = self.get_game_data(game_id)
                games.append(game_data)

        print(f"Found {len(games)} games of type {game_types} for season {season}")

        return games

    def get_game_data(self, game_id: str) -> object:
        """
        Get data from API or from cache is cache storage is provided
        """
        uri = f"/gamecenter/{game_id}/play-by-play"
        return self.fetch_from_url_and_cache(self.games_base_url, uri, "games/")

    def get_game_numbers_in_season(self, season: int, game_type: str) -> None | list[int]:
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
            return None

        if game_type == GameType.REGULAR:
            max_number = int(season_data["totalRegularSeasonGames"])
            return list(range(1, max_number + 1))
        elif game_type == GameType.PLAYOFF:
            return self.get_playoff_games_number(season)
        else:
            raise Exception("Game type '%s' not recognized" % game_type)

    def get_playoff_games_number(self, season: int) -> list[int]:
        """
        Get the number of playoff games in a season
        Based on this endpoint: https://api-web.nhle.com/v1/playoff-series/carousel/20232024/
        Add wins of bottomSeed and topSeed to get the number of games played in a specific series
        """
        playoff_series = self.get_playoff_series(season)

        def get_game_count_for_series(round: int, series: int) -> int:
            """
            Get the number of games for a specific series
            """
            # Compute the letter of the series
            series_letter = get_series_letter(round, series)
            # Found in https://www.geeksforgeeks.org/python-find-dictionary-matching-value-in-list/
            r = next((r for r in playoff_series["rounds"] if r["roundNumber"] == round), None)
            s = next((s for s in r["series"] if s["seriesLetter"] == series_letter), None)

            # Add the number of wins for the top and bottom
            return s["bottomSeed"]["wins"] + s["topSeed"]["wins"]

        numbers = list()

        # TODO: Could be done with numpy ?
        for round in range(1,5):
            series_count = 2 ** (4 - round)
            for series in range(1, series_count+1):
                games_count = get_game_count_for_series(round, series)
                for game in range(1, games_count+1):
                    numbers.append(int(f"{round}{series}{game}"))

        return numbers

    def get_playoff_series(self, season: int) -> dict[str | int] | None:
        """
        Get the playoff brackets details for a given season
        https://api-web.nhle.com/v1/playoff-series/carousel/20232024/
        """
        uri = "/playoff-series/carousel/" + str(season) + str(season + 1) + "/"
        return self.fetch_from_url_and_cache(self.games_base_url, uri, "games/")

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
        return self.fetch_from_url_and_cache(self.stats_base_url, "/season", "stats/")["data"]

    def fetch_from_url_and_cache(self, base_url: str, uri: str, cache_prefix: str) -> dict[str | int] | None:
        """
        Fetch data from an API URL and cache it
        """
        cache_key = cache_prefix + uri

        # Try to get result from cache
        if self.cache is not None:
            value = self.cache.get(cache_key)
            if value is not None:
                print(f"[ApiClient.fetch_from_url_and_cache] Loaded '{uri}' from cache")
                return json.loads(value)

        # Get content from API or raise an error
        full_url = base_url + uri
        response = requests.get(full_url)
        response.raise_for_status()
        text = response.text

        print(f"[ApiClient.fetch_from_url_and_cache] Loaded '{uri}' from API")

        # Store the content if cache is enabled
        if self.cache is not None:
            self.cache.set(cache_key, text)

        return json.loads(text)