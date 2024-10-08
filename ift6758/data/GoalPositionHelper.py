import math

goal_x_coord = 5 # Todo: Find the real value
goal_y_coord = 0 # Todo: Find the real value

# Helper from https://www.geeksforgeeks.org/how-to-compute-the-angle-between-vectors-using-python/
def angle_between_vectors(u, v):
    dot_product = sum(i*j for i, j in zip(u, v))
    norm_u = math.sqrt(sum(i**2 for i in u))
    norm_v = math.sqrt(sum(i**2 for i in v))
    cos_theta = dot_product / (norm_u * norm_v)
    angle_rad = math.acos(cos_theta)
    angle_deg = math.degrees(angle_rad)
    return angle_rad, angle_deg

def distance_between_points(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def invert_side(side: str) -> str:
    """
    Invert the ice side
    """
    return "left" if side == "right" else "right"


class GoalPositionHelper:

    def __init__(self, game_data: dict):
        self.game_data = game_data

        self.away_team = self.game_data.get("awayTeam", {}).get("id")
        if self.away_team is None:
            raise ValueError("awayTeam is missing")

        self.home_team = self.game_data.get("homeTeam", {}).get("id")
        if self.home_team is None:
            raise ValueError("homeTeam is missing")

        self.home_team_side = self.guess_ice_side_for_home_team_during_first_period()

    def distance_and_angle_to_adverse_goal(self, event: dict) -> (float, float):
        """
        Get the distance to the adverse goal for the event
        """
        x_goal, y_goal = self.get_adverse_goal_position(event)

        if x_goal is None or y_goal is None:
            return None

        details = event.get("details", {})
        x_event = details.get("xCoord")
        y_event = details.get("yCoord")

        distance = distance_between_points(x_event, y_event, x_goal, y_goal)

        # Vector from goal to event
        goal_to_event = (x_event - x_goal, y_event - y_goal)
        # Vector from goal to center
        goal_to_center = (0 - x_goal, 0 - y_goal)
        # Angle between the two vectors
        # 0 -> the event is in front of the goal
        # 90 -> the event is on the right of the goal
        angle_rad, angle_deg = angle_between_vectors(goal_to_event, goal_to_center)

        return distance, angle_deg

    def get_adverse_goal_position(self, event: dict) -> (int, int):
        """
        Get the position of the goal for the adverse team regarding the details.eventOwnerTeamId
        """
        # Quick if the home team size has been defined
        if self.home_team_side is None:
            return None

        details = event.get("details", {})
        event_owner_team_id = details.get("eventOwnerTeamId")
        period = event.get("periodDescriptor", {}).get("number")

        is_home_team = event_owner_team_id == self.home_team
        home_team_side = self.get_home_team_side_for_period(period)

        if is_home_team:
            if home_team_side == "left":
                return goal_x_coord, goal_y_coord
            else:
                return -goal_x_coord, goal_y_coord
        else:
            if home_team_side == "left":
                return -goal_x_coord, goal_y_coord
            else:
                return goal_x_coord, goal_y_coord

    def get_home_team_side_for_period(self, period: int) -> str:
        """
        Get the side of the home team for a given period
        """
        if self.home_team_side is None:
            raise ValueError("home_team_side is not defined")
        if period % 2 == 0:
            return invert_side(self.home_team_side)
        return self.home_team_side

    def guess_ice_side_for_home_team_during_first_period(self) -> str | None:
        first_event = self.find_first_relevant_event()
        if first_event is None:
            return None

        details = first_event.get("details", {})
        x = details.get("xCoord")
        zone = details.get("zoneCode")
        event_owner_team_id = details.get("eventOwnerTeamId")
        period = first_event.get("periodDescriptor", {}).get("number")

        is_home_team = event_owner_team_id == self.home_team

        if is_home_team:
            if zone == 'D': # Zone D for the home team -> Home team's side
                home_team_side = "left" if x < 0 else "right"
            else: # Zone O for the home team -> Away team's side
                home_team_side = "left" if x > 0 else "right"
        else:
            if zone == 'D': # Zone D for the away team -> Away team's side
                home_team_side = "left" if x > 0 else "right"
            else: # Zone O for the away team -> Home team's side
                home_team_side = "left" if x < 0 else "right"

        # Now we have the side of the home team during the event period.
        # We need to guess for the first period.
        # If the period is odd we invert the side
        if period % 2 == 0:
            return invert_side(home_team_side)
        else:
            return home_team_side

    def find_first_relevant_event(self) -> dict | None:
        """
        This method will look for the first event in the "play" list that has details.zoneCode `O` or `D`.
        From there, thanks to details.eventOwnerTeamId, details.xCoord, details.yCoord and periodDescriptor.number
        we can determine on which side of the ice the team is playing.
        """
        events = self.game_data.get("plays", [])
        for event in events:
            details = event.get("details", {})
            if details.get("zoneCode") in ['O', 'D']:
                return event
        return None


"""
GoalPositionHelper Factory
"""
get_goal_position_helpers_cache = dict()
def get_goal_position_helper(game_data: dict) -> GoalPositionHelper:
    """
    Get a GoalPositionHelper instance for a given game data
    Avoid creating multiple instances for the same game
    """
    game_id = game_data.get("id")
    if game_id in get_goal_position_helpers_cache:
        return get_goal_position_helpers_cache[game_id]
    else:
        helper = GoalPositionHelper(game_data)
        helper.analyse_game_data()
        get_goal_position_helpers_cache[game_id] = helper
        return helper