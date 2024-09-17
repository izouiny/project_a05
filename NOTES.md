# Notes taken during the project

## 1. Data fetching

### API Documentation

The [documentation linked](https://gitlab.com/dword4/nhlapi/-/blob/master/stats-api.md#game-ids) in the instructions are deprecated.
The API endpoint is not available anymore.

I found the newest [version of the documentation](https://gitlab.com/dword4/nhlapi/-/blob/master/new-api.md).

[This one](https://github.com/Zmalski/NHL-API-Reference) is also helpful.

### Helpers

#### Game ID

Addition of a helper function `get_game_id` to compute the game Id from date, game type and game number.

### Enums

Use of enums to define the game types.

### Data fetching & cache

Creation of a ApiClient class to fetch data from the NHL API.
This class accepts a `Cache` instance in its constructor to store the fetched data.
This allows to avoid fetching the same data multiple times.

`Cache` is an abstract class that defines the interface for a cache.
It defines the following methods:

- `has`
- `get`
- `set`
- `remove`
- `clear`

Creation of `FileSystemCache` class to store the fetched data in a local file. It implements the `Cache` interface.

We could imagine other Cache systems like a `RedisCache`, a `InMemoryCache`, etc.
But for this project, the `FileSystemCache` is enough.

### Usage of stats API to dinamically retrieve the game count in a season

Here is a sample response from `https://api.nhle.com/stats/rest/en/season`

```json
{
  "data": [
    {
      "id": 20232024,
      "allStarGameInUse": 1,
      "conferencesInUse": 1,
      "divisionsInUse": 1,
      "endDate": "2024-06-24T00:00:00",
      "entryDraftInUse": 1,
      "formattedSeasonId": "2023-24",
      "minimumPlayoffMinutesForGoalieStatsLeaders": 240,
      "minimumRegularGamesForGoalieStatsLeaders": 25,
      "nhlStanleyCupOwner": 1,
      "numberOfGames": 82,
      "olympicsParticipation": 0,
      "pointForOTLossInUse": 1,
      "preseasonStartdate": "2023-09-23T00:05:00",
      "regularSeasonEndDate": "2024-04-18T22:30:00",
      "rowInUse": 1,
      "seasonOrdinal": 106,
      "startDate": "2023-10-10T17:30:00",
      "supplementalDraftInUse": 0,
      "tiesInUse": 0,
      "totalPlayoffGames": 88,
      "totalRegularSeasonGames": 1312,
      "wildcardInUse": 1
    }
  ],
  "total": 107
}
```

### Game data for a whole season

Creation of a method `get_games_data` in the `ApiClient` class to fetch the data for a whole season.
This stores every game data in the cache.

**Issue**: The game number `0001` for playoffs seems to be not found.
https://api-web.nhle.com/v1/gamecenter/2020030001/play-by-play returns a 404.

**Solution**: Response found in [the documentation](https://gitlab.com/dword4/nhlapi/-/blob/master/stats-api.md#game-ids): 

> For playoff games, the 2nd digit of the specific number gives the round of the playoffs,
> the 3rd digit specifies the matchup, and the 4th digit specifies the game (out of 7).