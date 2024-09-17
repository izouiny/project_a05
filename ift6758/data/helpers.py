
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
