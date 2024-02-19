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
            "lr": {"type": "float", "min": 0, "max": 1, "default": 0.01},
            "momentum": {"type": "float", "min": 0, "max": 1, "default": 0.0}
        },
    },
    "FedProx": {
        "type": "dict",
        "default": {},
        "schema": {
            "mu": {"type": "float", "min": 0, "max": 1, "default": 0.01},
            "lr": {"type": "float", "min": 0, "max": 1, "default": 0.01},
            "momentum": {"type": "float", "min": 0, "max": 1, "default": 0.0}
        },
    },
    "FedNova": {
        "type": "dict",
        "default": {},
        "schema": {
            "lr": {"type": "float", "min": 0, "max": 1, "default": 0.01},
            "gmf": {"type": "float", "min": 0, "max": 1, "default": 0.9},
            "mu": {"type": "float", "min": 0, "max": 1, "default": 0.0},
            "momentum": {"type": "float", "min": 0, "max": 1, "default": 0.0}
        },
    },
    "FedDisco": {
        "type": "dict",
        "default": {},
        "schema": {
            "lr": {"type": "float", "min": 0, "max": 1, "default": 0.01},
            "mu": {"type": "float", "min": 0, "max": 1, "default": 0.0},
            "momentum": {"type": "float", "min": 0, "max": 1, "default": 0.0},
            "a": {"type": "float", "min": 0, "max": 2, "default": 0.1},
            "b": {"type": "float", "min": 0, "max": 1, "default": 0.03}

        },
    },
    "FedAvgM": {
        "type": "dict",
                "default": {},
                "schema": {
                    "lr": {"type": "float", "min": 0, "max": 1, "default": 0.01},
                    "gmf": {"type": "float", "min": 0, "max": 1, "default": 0.7},
                    "momentum": {"type": "float", "min": 0, "max": 1, "default": 0.0}
                }
    }
}
