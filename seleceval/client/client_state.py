from typing import Union, Dict

import pandas as pd


class ClientState:
    def __init__(self, cid: str, file: str):
        self.file = file
        state_df = pd.read_csv(self.file)
        self.state = state_df.to_dict(orient='records')[int(cid)]
        self.cid = cid

    def get(self, attr) -> Union[str, int, float]:
        return self.state[attr]

    def get_all(self) -> Dict:
        return self.state
