from typing import Union, Dict

import pandas as pd


class ClientState:
    def __init__(self, cid: str):
        state_df = pd.read_csv("client_states.csv")
        self.state = state_df.to_dict(orient='records')[int(cid)]
        self.cid = cid

    def get(self, attr) -> Union[str, int, float]:
        return self.state[attr]

    def set(self, attr, value) -> None:
        self.state[attr] = value

    def commit(self) -> None:
        state_df = pd.read_csv("client_states.csv")
        state_df.loc[int(self.cid), self.state.keys()] = self.state.values()
        state_df.to_csv("client_states.csv", index=False)

    def get_all(self) -> Dict:
        return self.state
