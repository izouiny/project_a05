# Notes taken during the project

## 1. Data fetching

### API Documentation

The [documentation linked](https://gitlab.com/dword4/nhlapi/-/blob/master/stats-api.md#game-ids) in the instructions are deprecated.
The API endpoint is not available anymore.

I found the newest [version of the documentation](https://gitlab.com/dword4/nhlapi/-/blob/master/new-api.md).

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


