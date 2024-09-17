from enum import Enum

class GameType(str, Enum):
    """01 = preseason, 02 = regular season, 03 = playoffs, 04 = all-star"""
    PRESEASON = "01"
    REGULAR = "02"
    PLAYOFF = "03"
    ALL_STAR = "04"
