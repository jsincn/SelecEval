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
            "lr": {"type": "float", "min": 0, "max": 1, "default": 0.02},
            "momentum": {"type": "float", "min": 0, "max": 1, "default": 0.0}
        },
    },
    "FedProx": {
        "type": "dict",
        "default": {},
        "schema": {
            "mu": {"type": "float", "min": 0, "max": 1, "default": 0.005},
            "lr": {"type": "float", "min": 0, "max": 1, "default": 0.02},
            "momentum": {"type": "float", "min": 0, "max": 1, "default": 0.9}
        },
    },
    "FedNova": {
        "type": "dict",
        "default": {},
        "schema": {
            "lr": {"type": "float", "min": 0, "max": 1, "default": 0.02},
            "gmf": {"type": "float", "min": 0, "max": 1, "default": 0.9},
            "mu": {"type": "float", "min": 0, "max": 1, "default": 0.005},
            "momentum": {"type": "float", "min": 0, "max": 1, "default": 0.9}
        },
    },
    "FedDisco": {
        "type": "dict",
        "default": {},
        "schema": {
            "lr": {"type": "float", "min": 0, "max": 1, "default": 0.02},
            "gmf": {"type": "float", "min": 0, "max": 1, "default": 0.9},
            "mu": {"type": "float", "min": 0, "max": 1, "default": 0.005},
            "momentum": {"type": "float", "min": 0, "max": 1, "default": 0.9}

        },
    },
    "FedAvgM": {
        "type": "dict",
                "default": {},
                "schema": {
                    "lr": {"type": "float", "min": 0, "max": 1, "default": 0.02},
                    "gmf": {"type": "float", "min": 0, "max": 1, "default": 0.9},
                    "momentum": {"type": "float", "min": 0, "max": 1, "default": 0.9}
                }
    }
}
