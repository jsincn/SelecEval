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

    def set(self, attr, value) -> None:
        self.state[attr] = value

    def commit(self) -> None:
        state_df = pd.read_csv(self.file)
        state_df.loc[int(self.cid), self.state.keys()] = self.state.values()
        state_df.to_csv(self.file, index=False)

    def get_all(self) -> Dict:
        return self.state
