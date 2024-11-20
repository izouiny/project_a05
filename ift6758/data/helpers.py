import pandas as pd

from .ApiClient import ApiClient
from .DataTransformer import DataTransformer, default_play_types
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

def load_events_records(season: int | None = None, all_types = False) -> list[dict]:
    """
    Loads raw data and flatten plays as json records
    """
    play_types = None if all_types else default_play_types
    raw_data = load_raw_games_data(season)

    return data_transformer.flatten_raw_data_as_records(raw_data, play_types)

def load_events_dataframe(season: int | None = None, all_types = False) -> pd.DataFrame:
    """
    Loads raw data and flatten plays
    """
    play_types = None if all_types else default_play_types
    raw_data = load_raw_games_data(season)

    return data_transformer.flatten_raw_data_as_dataframe(raw_data, play_types)

def load_train_test_dataframes(all_types = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train: seasons 2016 to 2019
    Test: season 2020
    """
    df_2016 = load_events_dataframe(2016, all_types=all_types)
    df_2017 = load_events_dataframe(2017, all_types=all_types)
    df_2018 = load_events_dataframe(2018, all_types=all_types)
    df_2019 = load_events_dataframe(2019, all_types=all_types)
    df_2020 = load_events_dataframe(2020, all_types=all_types).drop(columns="is_goal")

    return pd.concat([df_2016, df_2017, df_2018, df_2019]), df_2020
# -----------------------------------------------------------