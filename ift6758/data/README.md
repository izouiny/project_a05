# Data

This package is responsible for fetching and storing the data from the NHL API.

It uses the `ApiClient` class to fetch the data and the `FileSystemCache` class to store it.

After fetching the data, it merges all games and save the raw list of games in a file (one file per season).

## Examples

### Basic usage

```python
from ift6758.data import get_all_seasons_games_data

data = get_all_seasons_games_data()
```

Run the following command to clear the cache:

```python
from ift6758.data import clear_cache

clear_cache()
```

### Advanced usage

```python
from ift6758.data import (ApiClient, FileSystemCache, GameType)
import os

cache_path = os.path.dirname(os.path.abspath(__file__)) + "/storage/cache"
cache = FileSystemCache(cache_path)
client = ApiClient(cache)

data = client.get_games_data(2020, [GameType.REGULAR, GameType.PLAYOFF])
```
