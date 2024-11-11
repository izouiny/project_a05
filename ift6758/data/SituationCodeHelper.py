class SituationCodeHelper:

    def __init__(self, game_data: dict):
        self.game_data = game_data

        self.away_team = self.game_data.get("awayTeam", {}).get("id")
        if self.away_team is None:
            raise ValueError("awayTeam is missing")

        self.home_team = self.game_data.get("homeTeam", {}).get("id")
        if self.home_team is None:
            raise ValueError("homeTeam is missing")

    def is_adverse_net_empty(self, event: dict) -> bool:
        """
        Denotes if the goal was away from the net
        """
        situation_code = event.get("situationCode")

        if situation_code is None:
            return False

        details = event.get("details", {})
        team_id = details.get("eventOwnerTeamId")

        if team_id is None:
            return False

        if team_id == self.home_team:
            return situation_code[-1] == "0"
        else:
            return situation_code[0] == "0"

"""
SituationCodeHelper Factory
"""
get_situation_code_helpers_cache = dict()
def get_situation_code_helper(game_data: dict) -> SituationCodeHelper:
    """
    Get a GoalPositionHelper instance for a given game data
    Avoid creating multiple instances for the same game
    """
    game_id = game_data.get("id")
    if game_id in get_situation_code_helpers_cache:
        # print(f"Cache hit for game {game_id}")
        return get_situation_code_helpers_cache[game_id]
    else:
        # print(f"Cache miss for game {game_id}")
        helper = SituationCodeHelper(game_data)
        get_situation_code_helpers_cache[game_id] = helper
        return helper

# Unit Tests
if __name__ == "__main__":
    game_data = {
        "id": 1,
        "awayTeam": {"id": 10},
        "homeTeam": {"id": 20},
        "plays": [
            {"situationCode": "1551", "details": {"eventOwnerTeamId": 10}},
            {"situationCode": "1551", "details": {"eventOwnerTeamId": 20}},
            {"situationCode": "0551", "details": {"eventOwnerTeamId": 10}},
            {"situationCode": "1550", "details": {"eventOwnerTeamId": 20}},
        ]
    }
    helper = get_situation_code_helper(game_data)

    events = game_data["plays"]
    for event in events:
        print("-----------------------------")
        print(f"Team: {event['details']['eventOwnerTeamId']}, Situation Code: {event['situationCode']}")
        print("Is Adverse net empty?", helper.is_adverse_net_empty(event))
