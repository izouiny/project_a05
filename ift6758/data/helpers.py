import pandas as pd

from .ApiClient import ApiClient
from .DataTransformer import DataTransformer
from .FileSystemCache import FileSystemCache
from .enums import GameType
import os
import json

# -----------------------------------------------------------
# GLOBAL VARIABLES
cache_path = os.environ.get(
    "CACHE_PATH",
    os.path.dirname(os.path.abspath(__file__)) + "/storage/cache"
)
cache = FileSystemCache(cache_path)

dump_path = os.environ.get(
    "DUMP_PATH",
    os.path.dirname(os.path.abspath(__file__)) + "/storage/dump"
)
dump = FileSystemCache(dump_path)

api_client = ApiClient(cache)

data_transformer = DataTransformer()

all_seasons = range(2016, 2024)
game_types = [GameType.REGULAR, GameType.PLAYOFF]
# -----------------------------------------------------------


# -----------------------------------------------------------
# MAIN FUNCTIONS
def fetch_all_seasons_games_data() -> None:
    """
    Fetch all seasons games data from API or cache and update the dumps
    """
    for season in all_seasons:
        data = api_client.get_games_data(season, game_types)
        dump.set(f"{season}", json.dumps(data, indent=2))
        print(f"Stored dump for season {season}")

def clear_cache() -> None:
    """
    Clear the cache, not the dumps
    """
    cache.clear()

def load_raw_games_data(season: int | None = None) -> list[dict]:
    """
    This method load data from one specific season or all seasons
    """
    # Load all seasons
    if season is None:
        output = list()
        for s in all_seasons:
            data = dump.get(f"{s}")
            if data is not None:
                output = output + json.loads(data)
        return output

    # Load one specific season
    data = dump.get(f"{season}")
    if data is None:
        return []
    return json.loads(data)

def load_events_dataframe(season: int | None = None) -> pd.DataFrame:
    """
    Loads raw data and flatten plays
    """
    raw_data = load_raw_games_data(season)

    return data_transformer.flatten_raw_data(raw_data)
# -----------------------------------------------------------