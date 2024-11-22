from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, Normalizer, StandardScaler

from .ColumnDropperTransformer import ColumnDropperTransformer

def get_preprocessing_pipeline() -> Pipeline:
    """
    Returns a scikit-learn data pipeline that can be used to transform data
    Compatible with scikit-learn pipelines
    """

    #-------------------------------------------
    # Create features sets

    dropped_features = [
        'game_id',
        'season',
        'game_date',
        'venue',
        'venue_location',
        'away_team_id',
        'away_team_abbrev',
        'away_team_name',
        'home_team_id',
        'home_team_abbrev',
        'home_team_name',
        'event_id',
        'event_idx',
        'sort_order',
        'time_in_period',
        'time_remaining',
        'description',
        'event_owner_team_id',
        'details_type_code',
        'shooting_player_name',
        'shooting_player_team_id',
        'goalie_in_net_name',
        'goalie_in_net_team_id',

        # Remove source columns that are not useful anymore
        'x_coord',
        'y_coord',
        'goal_x_coord',
        'last_x',
        'last_y',
        'goal_side',

        # Remove target column and related columns
        'is_goal',
        'type_desc_key',
        'scoring_player_id',
        'scoring_player_name',
        'scoring_player_team_id',
        'scoring_player_position_code',
        'assist1_player_id',
        'assist1_player_name',
        'assist1_player_team_id',
        'assist1_player_position_code',
        'assist2_player_id',
        'assist2_player_name',
        'assist2_player_team_id',
        'assist2_player_position_code',
        'away_score',
        'home_score',

        # To check if we can remove these columns
        'away_sog', # We don't know if the event is from home or away team ?
        'home_sog', # We don't know if the event is from home or away team ?
        'shooting_team_skaters',
        'opposing_team_skaters',
    ]

    categorical_features = [
        'period_type',
        'zone_code',
        'shot_type',
        'goalie_in_net_position_code',
        'shooting_player_position_code',
        'last_event_type',
    ]

    distributed_features = [
        'goal_distance',
        'goal_angle',
        # 'time_since_last_event',
        # 'game_seconds',
        'distance_from_last_event',
        'speed',
        'last_angle',
        'absolute_angle_change',
        # 'power_play_time_elapsed',
    ]


    #-------------------------------------------
    # Create per feature type transformers

    numeric_transformer = Pipeline([
        # Replace missing values
        ('imputer', SimpleImputer(strategy='mean')),

        # Normalize features
        ('norm', Normalizer()),
        # ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        # One hot encode categorical features
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    #-------------------------------------------
    # Create final pipeline

    pipeline = Pipeline([
        # Drop unused columns
        ('drop_columns', ColumnDropperTransformer(dropped_features)),

        # Preprocess numeric and categorical features differently
        ('col_transformer', ColumnTransformer([
            ("cat", categorical_transformer, categorical_features),
            ("dist", numeric_transformer, distributed_features),
        ])),
    ])

    return pipeline
