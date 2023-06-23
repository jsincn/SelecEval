import json


class Config:
    def __init__(self, file_name: str):
        self.current_round = 0
        with open(file_name) as json_file:
            self.initial_config = json.load(json_file)

    def set_current_round(self, i: int):
        self.current_round = i

    def get_current_round(self):
        return self.current_round
