"""
Client state class
"""
from typing import Union, Dict

import pandas as pd


class ClientState:
    """
    Utility class for handling client state
    """

    def __init__(self, cid: str, file: str):
        self.file = file
        state_df = pd.read_csv(self.file)
        self.state = state_df.to_dict(orient='records')[int(cid)]
        self.cid = cid

    def get(self, attr) -> Union[str, int, float]:
        """
        Get attribute from state
        :param attr: Attribute to get
        :return: Value of attribute
        """
        return self.state[attr]

    def get_all(self) -> Dict:
        """
        Get all attributes from state
        :return: Dictionary of all attributes
        """
        return self.state
