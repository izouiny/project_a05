from enum import Enum

class GameType(str, Enum):
    PRESEASON = "01"
    REGULAR = "02"
    PLAYOFF = "03"
    ALL_STAR = "04"
