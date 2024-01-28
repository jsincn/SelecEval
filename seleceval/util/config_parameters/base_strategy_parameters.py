"""
This file contains the available strategies and the default strategy.
"""
available_strategies = ["FedAvgM", "FedAvg", "FedMedian", "FedProx", "FedNova", "FedDisco"]

default_strategy = "FedAvg"

base_strategy_parameter_dict = {
    "FedAvg": {
        "type": "dict",
        "default": {},
        "schema": {
            "lr": {"type": "float", "min": 0, "max": 1, "default": 0.001},
        },
    },
    "FedProx": {
        "type": "dict",
        "default": {},
        "schema": {
            "mu": {"type": "float", "min": 0, "max": 1, "default": 0.01},
            "lr": {"type": "float", "min": 0, "max": 1, "default": 0.001},
        },
    },
    "FedNova": {
        "type": "dict",
        "default": {},
        "schema": {
            "lr": {"type": "float", "min": 0, "max": 1, "default": 0.01},
            "gmf": {"type": "float", "min": 0, "max": 1, "default": 0.00},
            "mu": {"type": "float", "min": 0, "max": 1, "default": 0.01},
        },
    },
    "FedDisco": {
        "type": "dict",
        "default": {},
        "schema": {
            "lr": {"type": "float", "min": 0, "max": 1, "default": 0.01},
            "gmf": {"type": "float", "min": 0, "max": 1, "default": 0},
            "mu": {"type": "float", "min": 0, "max": 1, "default": 0.01},
        },
    },
}
