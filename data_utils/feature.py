from collections import defaultdict

class Feature(object):
    def __init__(self, features: dict):
        self.__dict__ = defaultdict(lambda x: None)
        for key, value in features.items():
            self.__dict__[key] = value

    def data(self) -> dict:
        return self.__dict__