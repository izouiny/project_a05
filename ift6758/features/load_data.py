import pandas as pd
from sklearn.model_selection import train_test_split

from ift6758.data import load_events_dataframe
from .preprocess_advanced import preprocess_advanced
from .preprocessing_pipeline import features_to_drop

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
    df_2020 = load_advanced_dataframe(2020)

    return pd.concat([df_2016, df_2017, df_2018, df_2019]), df_2020

def load_train_val_test_x_y(test_size: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load train, validation, and test dataframes
    """
    train_df, test_df = load_advanced_train_test_dataframes()

    # Split train into train and validation
    train_df, val_df = train_test_split(train_df, test_size=test_size, random_state=42)

    # Target column
    target_col = "is_goal"

    # Split X and y
    X_train = train_df.drop(features_to_drop, axis=1, errors="ignore")
    y_train = train_df[target_col]

    X_val = val_df.drop(features_to_drop, axis=1, errors="ignore")
    y_val = val_df[target_col]

    X_test = test_df.drop(features_to_drop, axis=1, errors="ignore")
    y_test = test_df[target_col]

    return X_train, y_train, X_val, y_val, X_test, y_test