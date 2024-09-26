import pandas as pd

class DataTransformer:

    """
    This class transforms a list of games in JSON into Panda dataframe
    """

    play_types = ("shot-on-goal", "goal")


    def flatten_raw_data_as_dataframe(self, games: list[dict]) -> pd.DataFrame :
        """
        Convert records into a dataframe
        """
        events = self.flatten_raw_data_as_records(games)

        return pd.DataFrame.from_records(events)

    def flatten_raw_data_as_records(self, games: list[dict]) -> list[dict] :
        """
        Get data from an entire season.
        This methods will request the API a lot of time.
        """
        events = list()

        for game in games:
            rows = self.flatten_game(game)
            events.extend(rows)

        print(f"Found {len(events)} events")

        return events

    def flatten_game(self, game_data: dict) -> list[dict]:
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

        for play in plays:
            # Get play type and quit if not in the list
            type = play.get("typeDescKey")
            if type not in self.play_types:
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

            # Get the root properties
            period = play.get("periodDescriptor", {})
            event_props = {
                'event_id': play.get("eventId"),
                'period_number': period.get("number"),
                'period_type': period.get("periodType"),
                'time_in_period': play.get("timeInPeriod"),
                'time_remaining': play.get("timeRemaining"),
                'situation_code': play.get("situationCode"),
                'type_code': play.get("typeCode"),
                'type_desc_key': play.get("typeDescKey"),
                'sort_order': play.get("sortOrder"),

                # Details
                'x_coord': details.get("xCoord"),
                'y_coord': details.get("yCoord"),
                'zone_code': details.get("zoneCode"),
                'shot_type': details.get("shotType"),
                'event_owner_team_id': details.get("eventOwnerTeamId"),
            }

            events.append({
                **root_props,
                **event_props,
                **goalie_in_net,
                **shooting_player,
                **scoring_player,
                **assist1_player,
                **assist2_player,
            })

        return events


