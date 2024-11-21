import pandas as pd
from ift6758.data import load_events_dataframe
from .preprocess_advanced import preprocess_advanced

def load_advanced_dataframe(season: int | None = None) -> pd.DataFrame:
    """
    Loads events dataframe and add advanced features
    """
    # Load raw data
    raw_data = load_events_dataframe(season, all_types=True)

    # Sort data by game_id, period_number, and time_in_period
    raw_data = raw_data.sort_values(by=["game_id", "period_number", "time_in_period"])

    return preprocess_advanced(raw_data)

def load_advanced_train_test_dataframes() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train: seasons 2016 to 2019
    Test: season 2020
    """
    df_2016 = load_advanced_dataframe(2016)
    df_2017 = load_advanced_dataframe(2017)
    df_2018 = load_advanced_dataframe(2018)
    df_2019 = load_advanced_dataframe(2019)
    df_2020 = load_advanced_dataframe(2020).drop(columns="is_goal")

    return pd.concat([df_2016, df_2017, df_2018, df_2019]), df_2020
