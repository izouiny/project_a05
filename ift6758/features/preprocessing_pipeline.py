from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, Normalizer, StandardScaler

from .ColumnDropperTransformer import ColumnDropperTransformer


def get_preprocessing_pipeline() -> Pipeline:
    """
    Returns a scikit-learn data pipeline that can be used to transform data
    Compatible with scikit-learn pipelines
    """
    pipeline = Pipeline([
        # Drop unused columns
        ('drop_columns', ColumnDropperTransformer([
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
            'goal_x_coord',
            'shooting_player_name',
            'shooting_player_team_id',
            'goalie_in_net_name',
            'goalie_in_net_team_id',
            'scoring_player_name',
            'scoring_player_team_id',
            'assist1_player_name',
            'assist1_player_team_id',
            'assist2_player_name',
            'assist2_player_team_id',

            # Remove target columns
            'is_goal',
            'type_desc_key',

            # To check if we can remove these columns
            'away_score', # Almost always NaN
            'home_score', # Almost always NaN
            'shooting_team_skaters',
            'opposing_team_skaters',
        ])),

        # One hot encode categorical features
        ('ohe', OneHotEncoder()),

        # Replace missing values
        ('imputer', SimpleImputer(strategy='mean')),

        # Normalize features
        # ('norm', Normalizer()),
        # ('scaler', StandardScaler())
    ])

    return pipeline
