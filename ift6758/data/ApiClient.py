import requests

class ApiClient:

    """
    This class allows to interact with the API of the NHL
    """

    base_url = "https://api-web.nhle.com/v1"

    def __init__(self, use_storage: bool = True):
        """
        Args:
            use_storage: Define whether to save and load data from storage (when applicable). This limits call done to the API.
        """
        self.use_storage = use_storage

    def get_game_id(self, season: int, game_type: str, game_number: int) -> str:
        """
        Compute the game id from

        Args:
            season (int): the season for which to retrieve the game ID (use starting year of the season)
            game_type (int): type of game
            game_number (int): identify the specific game number.

        Returns:
            list: The game id as string
        """
        return str(season) + game_type + ('%04d' % game_number)

