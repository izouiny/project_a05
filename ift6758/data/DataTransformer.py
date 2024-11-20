import pandas as pd
from .GoalPositionHelper import get_goal_position_helper
from .SituationCodeHelper import get_situation_code_helper

default_play_types = ("shot-on-goal", "goal")

class DataTransformer:

    """
    This class transforms a list of games in JSON into Panda dataframe
    """

    def flatten_raw_data_as_dataframe(self, games: list[dict], play_types: tuple[str] | None = default_play_types) -> pd.DataFrame :
        """
        Convert records into a dataframe
        """
        events = self.flatten_raw_data_as_records(games, play_types)

        return pd.DataFrame.from_records(events)

    def flatten_raw_data_as_records(self, games: list[dict], play_types: tuple[str] | None = default_play_types) -> list[dict] :
        """
        Get data from an entire season.
        This methods will request the API a lot of time.
        """
        events = list()

        for game in games:
            rows = self.flatten_game(game, play_types)
            events.extend(rows)

        print(f"Found {len(events)} events")

        return events

    def flatten_game(self, game_data: dict, play_types: tuple[str] | None) -> list[dict]:
        """
        This method flatten a single game
        """

        # Get the root properties
        away_team = game_data.get("awayTeam", {})
        home_team = game_data.get("homeTeam", {})
        root_props = {
            'game_id': game_data.get("id"),
            'season': game_data.get("season"),
            'game_type': game_data.get("gameType"),
            'game_date': game_data.get("gameDate"),
            'venue': game_data.get("venue", {}).get("default"),
            'venue_location': game_data.get("venueLocation", {}).get("default"),

            'away_team_id': away_team.get("id"),
            'away_team_abbrev': away_team.get("abbrev"),
            'away_team_name': away_team.get("name", {}).get("default"),

            'home_team_id': home_team.get("id"),
            'home_team_abbrev': home_team.get("abbrev"),
            'home_team_name': home_team.get("name").get("default"),
        }

        # Get the players list
        # https://www.quora.com/How-do-I-convert-a-list-into-objects-in-Python
        players_list = game_data.get("rosterSpots", [])
        players = {
            str(player.get("playerId")): {
                'name': player.get("firstName", {}).get("default") + " " + player.get("lastName", {}).get("default"),
                'team_id': player.get("teamId"),
                'position_code': player.get("positionCode"),
            }
            for player in players_list
        }
        # Helper function to get player details. This avoids calling the same code multiple times
        def player_details(player_id: int, prefix: str) -> dict:
            player = players.get(str(player_id))
            if player is None:
                return {}
            return {
                f'{prefix}_id': player_id,
                f'{prefix}_name': player.get("name"),
                f'{prefix}_team_id': player.get("team_id"),
                f'{prefix}_position_code': player.get("position_code"),
            }

        plays = game_data.get("plays", [])
        events = list()

        # Create a helper to get goal position details
        goal_position_helper = get_goal_position_helper(game_data)

        # Create a helper to get situation code details
        situation_code_helper = get_situation_code_helper(game_data)

        for index, play in enumerate(plays):
            # Get play type and quit if not in the list
            play_type = play.get("typeDescKey")
            if play_types is not None and play_type not in play_types:
                continue

            # Get details once
            details = play.get("details", {})

            # Get players properties
            # This avoids getting multiple times the same nested object
            goalie_in_net = player_details(details.get("goalieInNetId"), 'goalie_in_net')
            shooting_player = player_details(details.get("shootingPlayerId"), 'shooting_player')
            scoring_player = player_details(details.get("scoringPlayerId"), 'scoring_player')
            assist1_player = player_details(details.get("assist1PlayerId"), 'assist1_player')
            assist2_player = player_details(details.get("assist2PlayerId"), 'assist2_player')

            # Describe the event
            if play_type == "goal":
                description = f"{scoring_player.get('scoring_player_name')} scores a goal"
            elif play_type == "shot-on-goal":
                description = f"{goalie_in_net.get('goalie_in_net_name')} stops a shot from {shooting_player.get('shooting_player_name')}"
            else:
                description = f"Event {play_type}"

            # Get the root properties
            period = play.get("periodDescriptor", {})
            event_props = {
                'event_id': play.get("eventId"),
                'event_idx': index,
                'sort_order': play.get("sortOrder"),

                'period_number': period.get("number"),
                'period_type': period.get("periodType"),
                'max_regulation_periods': period.get("maxRegulationPeriods"),

                'time_in_period': play.get("timeInPeriod"),
                'time_remaining': play.get("timeRemaining"),
                'situation_code': play.get("situationCode"),

                'is_empty_net': situation_code_helper.is_adverse_net_empty(play),
                'is_goal': play_type == "goal",

                'type_code': play.get("typeCode"),
                'type_desc_key': play_type,

                'away_score': details.get("awayScore"),
                'home_score': details.get("homeScore"),

                'away_sog': details.get("awaySOG"),
                'home_sog': details.get("homeSOG"),

                # Details
                'x_coord': details.get("xCoord"),
                'y_coord': details.get("yCoord"),
                'zone_code': details.get("zoneCode"),
                'shot_type': details.get("shotType"),
                'description': description,
                'event_owner_team_id': details.get("eventOwnerTeamId"),

                # For penalties
                'details_type_code': details.get("typeCode"),

                'scoring_player_total': details.get("scoringPlayerTotal"),
                'assist1_player_total': details.get("assist1PlayerTotal"),
                'assist2_player_total': details.get("assist2PlayerTotal"),
            }

            # Get the goal position details
            goal_position_details = goal_position_helper.get_player_to_goal_details(play)

            events.append({
                **root_props,
                **event_props,
                **goalie_in_net,
                **shooting_player,
                **scoring_player,
                **assist1_player,
                **assist2_player,
                **goal_position_details,
            })

        return events


