# Notes taken during the project

## 1. Data fetching

### API Documentation

The [documentation linked](https://gitlab.com/dword4/nhlapi/-/blob/master/stats-api.md#game-ids) in the instructions are deprecated.
The API endpoint is not available anymore.

I found the newest [version of the documentation](https://gitlab.com/dword4/nhlapi/-/blob/master/new-api.md).

[This one](https://github.com/Zmalski/NHL-API-Reference) is also helpful.

### Helpers

#### Game ID & Series Letter

Addition of a helper function `get_game_id` to compute the game Id from date, game type and game number.

Addition of a helper function `get_series_letter` to compute the series letter from the round and games indices.

### Enums

Use of enums to define the game types.

### Data fetching & cache

Creation of a ApiClient class to fetch data from the NHL API.
This class accepts a `Cache` instance in its constructor to store the fetched data.
This allows to avoid fetching the same data multiple times.
This also allows to download the whole data in multiple steps.

#### Cache engine

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

##### Improvement

Add method `fetch_from_url_and_cache` in `ApiClient` to avoid repetition of the same code.

### Endpoints

#### Usage of stats API to dynamically retrieve the game count in a season

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

#### Game data for a whole season

Creation of a method `get_games_data` in the `ApiClient` class to fetch the data for a whole season.
This stores every game data in the cache.

**Issue**: The game number `0001` for playoffs seems to be not found.
https://api-web.nhle.com/v1/gamecenter/2020030001/play-by-play returns a 404.

**Solution**: Response found in [the documentation](https://gitlab.com/dword4/nhlapi/-/blob/master/stats-api.md#game-ids): 

> For playoff games, the 2nd digit of the specific number gives the round of the playoffs,
> the 3rd digit specifies the matchup, and the 4th digit specifies the game (out of 7).

1. Found information in this endpoint: https://api-web.nhle.com/v1/playoff-bracket/2022
2. This endpoint has some issues, series letter are not always the same and the sum of the wins mismatch the game numbers.
3. Found a better endpoint: https://api-web.nhle.com/v1/playoff-series/carousel/20232024/


#### Usage of the Playoff carrousel API to dynamically retrieve the games count in a specific series

Finding the number of games in a series was not trivial. This can be from 4 to 7 games.

Based on [this endpoint](https://api-web.nhle.com/v1/playoff-series/carousel/20232024/) we can find this information.
First, in order to find the right series, we have to define the letter of hte series from the round and the series index.
Then,  we had to add the wins of each team to find the number of games played.
From this, we can find the number of games left to play.
We are now able to generate all the game numbers for playoffs.
