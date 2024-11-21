import pandas as pd
import numpy as np

def convert_to_seconds(time_str):
    minutes, seconds = map(int, time_str.split(':'))
    return minutes * 60 + seconds


def add_previous_event_features(data):
    data["last_event_type"] = data["type_desc_key"].shift(1)
    data["last_x"] = data["x_coord"].shift(1)
    data["last_y"] = data["y_coord"].shift(1)
    data["time_since_last_event"] = data["game_seconds"] - data["game_seconds"].shift(1)

    def calculate_distance(row):
        if pd.isna(row["last_x"]) or pd.isna(row["last_y"]):
            return np.nan
        return np.sqrt((row["x_coord"] - row["last_x"]) ** 2 + (row["y_coord"] - row["last_y"]) ** 2)

    data["distance_from_last_event"] = data.apply(calculate_distance, axis=1)
    return data


def add_last_angle(data):
    data["last_angle"] = data["goal_angle"].shift(1)
    data.loc[data["game_id"] != data["game_id"].shift(1), "last_angle"] = np.nan
    return data


def calculate_absolute_angle_change(row):
    if not row["is_rebound"]:
        return 0
    return abs(row["goal_angle"]) + abs(row["last_angle"])


def calculate_speed(row):
    if pd.isna(row["time_since_last_event"]) or row["time_since_last_event"] <= 0:
        return 0
    return row["distance_from_last_event"] / row["time_since_last_event"]


def calculate_power_play_time(data):
    data = data.copy()
    data["power_play_time_elapsed"] = 0
    active_penalties = {"home": [], "away": []}

    for idx, row in data.iterrows():
        current_time = row["game_seconds"]
        for team in ["home", "away"]:
            active_penalties[team] = [
                (end_time, start_time) for end_time, start_time in active_penalties[team] if end_time > current_time
            ]
        if row["type_desc_key"] == "penalty":
            penalized_team = "home" if row["event_owner_team_id"] == row["away_team_id"] else "away"
            penalty_duration = 120 if row["details_type_code"] == "MIN" else 300
            active_penalties[penalized_team].append((current_time + penalty_duration, current_time))
        home_penalties = len(active_penalties["home"])
        away_penalties = len(active_penalties["away"])
        if home_penalties > away_penalties:
            power_play_time = current_time - min(start for _, start in active_penalties["home"])
        elif away_penalties > home_penalties:
            power_play_time = current_time - min(start for _, start in active_penalties["away"])
        else:
            power_play_time = 0
        power_play_time = max(0, power_play_time)
        data.at[idx, "power_play_time_elapsed"] = power_play_time
    return data


def add_skater_counts_for_shooting_team(data):
    def parse_skater_counts(row):
        try:
            home_skaters = int(str(row["situation_code"])[1])
            away_skaters = int(str(row["situation_code"])[2])
            if row["event_owner_team_id"] == row["home_team_id"]:
                return home_skaters, away_skaters
            else:
                return away_skaters, home_skaters
        except (ValueError, TypeError):
            return 5, 5

    skater_counts = data.apply(parse_skater_counts, axis=1)
    data["shooting_team_skaters"] = skater_counts.apply(lambda x: x[0])
    data["opposing_team_skaters"] = skater_counts.apply(lambda x: x[1])
    return data

def preprocess_advanced(data):
    data = data.copy()
    data['time_in_period_seconds'] = data['time_in_period'].apply(convert_to_seconds)
    data['game_seconds'] = (data['period_number'] - 1) * 1200 + data['time_in_period_seconds']
    data = add_previous_event_features(data)
    data["is_rebound"] = data["last_event_type"].isin(["shot-on-goal", "missed-shot", "blocked-shot"])
    data["speed"] = data.apply(calculate_speed, axis=1)
    data = add_last_angle(data)
    data["absolute_angle_change"] = data.apply(calculate_absolute_angle_change, axis=1)
    data = calculate_power_play_time(data)
    data = add_skater_counts_for_shooting_team(data)
    data = data[data["type_desc_key"].isin(["shot-on-goal", "goal"])]
    return data
