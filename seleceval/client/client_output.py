import json
import datetime
from typing import Any, Union

from .client_state import ClientState


class ClientOutput:

    def __init__(self, state: ClientState, server_round: int, file: str):
        self.file = file
        self.output_dict = {'server_round': server_round, 'client_name': state.get('client_name'), 'state': state.get_all()}

    def set(self, key: Union[str, int], value: Any):
        self.output_dict[key] = value

    def get(self, key: Union[str, int]) -> Any:
        return self.output_dict['key']

    def write(self):
        self.output_dict['current_timestamp'] = str(datetime.datetime.now())
        f = open(self.file, "a")
        f.write(json.dumps(self.output_dict) + "\n")
        f.close()
