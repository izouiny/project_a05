import os
import shutil

from .Cache import Cache


class FileSystemCache(Cache):

    """
    This class stores cache in local file system
    """

    def __init__(self, base_path: str):
        # Remove trailing slash
        # Seen here : https://stackoverflow.com/questions/15478127/remove-final-character-from-string
        self.base_path = base_path[:-1] if base_path.endswith("/") else base_path

    def has(self, key: str) -> bool:
        return os.path.isfile(self.get_file_path_for_key(key))

    def get(self, key: str) -> str | None:
        if self.has(key):
            # From https://stackoverflow.com/questions/3758147/easiest-way-to-read-write-a-files-content-in-python
            with open(self.get_file_path_for_key(key)) as file: content = file.read()
            return content
        else:
            return None

    def set(self, key: str, value: str) -> None:
        file_path = self.get_file_path_for_key(key)
        self.make_dirs_for_key(key)
        file = open(file_path, "w")
        file.write(value)
        file.close()

    def remove(self, key: str) -> bool:
        if self.has(key):
            os.remove(self.get_file_path_for_key(key))
            return True
        else:
            return False

    def clear(self) -> None:
        shutil.rmtree(self.base_path, ignore_errors=True)

    def get_file_path_for_key(self, key: str) -> str:
        return self.base_path + "/" + key + ".json"

    def make_dirs_for_key(self, key: str) -> None:
        file_path = self.get_file_path_for_key(key)
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)