"""
Utility class for handling client output
"""
import datetime
import fcntl
import json
from typing import Any, Union

from .client_state import ClientState


class ClientOutput:

    def __init__(self, state: ClientState, server_round: int, file: str):
        self.file = file
        self.output_dict = {'server_round': server_round, 'client_name': state.get('client_name'),
                            'state': state.get_all()}

    def set(self, key: Union[str, int], value: Any):
        """
        Set output key to values
        :param key: String or integer key
        :param value: Value to set
        """
        self.output_dict[key] = value

    def get(self, key: Union[str, int]) -> Any:
        """
        Get output value for key
        :param key: String or integer key
        :return: Value for key
        """
        return self.output_dict[key]

    def write(self):
        """
        Write output
        """
        self.output_dict['current_timestamp'] = str(datetime.datetime.now())
        with open(self.file, "a") as g:
            fcntl.flock(g, fcntl.LOCK_EX)
            g.write(json.dumps(self.output_dict) + "\n")
            fcntl.flock(g, fcntl.LOCK_UN)
