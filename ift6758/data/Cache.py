class Cache:
    """
    This is an interface for all cache engines.
    This is a storage based on key/value
    """

    def has(self, key: str) -> bool:
        """Must be implemented in subclass"""
        pass

    def get(self, key: str) -> str | None:
        """Must be implemented in subclass"""
        pass

    def set(self, key: str, value: str) -> None:
        """Must be implemented in subclass"""
        pass

    def remove(self, key: str) -> bool:
        """Must be implemented in subclass"""
        pass

    def clear(self) -> None:
        """Must be implemented in subclass"""
        pass

