import json
import sys

from runify.constants import *


class Localization:
    def __init__(self, language: Language = Language.ENGLISH, data: dict = None):
        self._language = language
        self._data = data if data is not None else self._load_data()

    def _load_data(self):
        # Load JSON file named after the language enum value
        file_name = LOCALIZATION_DATA_FOLDER + "/" + f"{self._language.value}.json"
        try:
            with open(file_name, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            if self.language != Language.ENGLISH:
                print(f"Failed to load localization file {file_name}. Using default language (english)")
                self._language = Language.ENGLISH
                return self._load_data()
            else:
                print("Failed to load default localization file. RIP")
                sys.exit(0)

    def __getattr__(self, key):
        # Handle top-level keys
        if key in self._data:
            value = self._data[key]
            if isinstance(value, dict):
                # Return the dictionary directly, but wrap in Localization for further nested access
                return Localization(self._language, value)
            return value
        raise AttributeError(f"'{key}' not found in localization data")

    def __call__(self):
        # Return the entire data structure when called
        return self._data

    def __repr__(self):
        # Return the raw dictionary when accessed directly
        return str(self._data)

