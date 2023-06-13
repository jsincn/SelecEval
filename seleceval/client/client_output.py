import json
import time
from typing import Any, Union

from seleceval.client.client_state import ClientState


class ClientOutput:

    def __init__(self, state: ClientState):
        self.output_dict = {'client_name': state.get('clientName'), 'state': state.get_all()}

    def set(self, key: Union[str, int], value: Any):
        self.output_dict[key] = value

    def get(self, key: Union[str, int]) -> Any:
        return self.output_dict['key']

    def write(self):
        self.output_dict['current_timestamp'] = time.time()
        f = open("output.txt", "a")
        f.write(json.dumps(self.output_dict) + "\n")
        f.close()
