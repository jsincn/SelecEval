"""
This file contains the available data quantity distributions as well as their parameters and the default.
"""

data_quantity_distributions = ["Dirichlet", "Uniform"]

data_quantity_distribution_parameters = {
    "data_quantity_base_parameter": {
        "type": "float",
        "min": 0.001,
        "max": 1,
        "default": 0.01,
    },
    "data_quantity_skew_parameter_1": {"type": "float", "min": 0, "default": 0.5},
    "data_quantity_skew_parameter_2": {"type": "float", "min": 0, "default": 0.5},
    "data_quantity_min_parameter": {"type": "integer", "min": 0, "default": 32},
    "data_quantity_max_parameter": {"type": "integer", "min": 1, "default": 2000}
}

default_quantity_distribution = "Uniform"
