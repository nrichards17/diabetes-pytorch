import json


class Params(dict):
    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self, f, indent=4)

    def load(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.update(params)


class Features(dict):
    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self, f, indent=4)

    def load(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.update(params)
