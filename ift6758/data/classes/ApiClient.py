import requests

class ApiClient:

    """
    This class allows to interact with the API of the NHL
    """

    base_url = "https://api-web.nhle.com/v1"

    def __init__(self, use_storage: bool = True):
        """
        Args:
            use_storage: Define whether to save and load data from storage (when applicable). This limits call done to the API.
        """
        self.use_storage = use_storage

