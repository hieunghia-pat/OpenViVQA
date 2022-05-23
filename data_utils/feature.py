from collections import defaultdict
from typing import Any

class Feature(object):
    def __init__(self, features: dict):
        for key, value in features.items():
            self.__dict__[key] = value