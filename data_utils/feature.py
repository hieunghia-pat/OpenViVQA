from collections import defaultdict

class Feature:
    def __init__(self, visual_features: dict):
        self.data = defaultdict(lambda x: None)
        for key, value in visual_features.items():
            self.data[key] = value

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()
    
    def items(self):
        return self.data.items()

    def __getattribute__(self, __name: str):
        return self.data[__name]
    