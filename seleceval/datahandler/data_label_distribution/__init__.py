"""
This module contains methods of skewing data labels
"""
from .discrete import Discrete
from .dirichlet import Dirichlet
from .uniform import Uniform

__all__ = [
    "Dirichlet",
    "Uniform",
    "Discrete",
    "data_label_distribution_dict",
    "data_label_distribution_parameters",
    "default_label_distribution",
]

data_label_distribution_dict = {
    "Uniform": Uniform,
    "Discrete": Discrete,
    "Dirichlet": Dirichlet,
}

data_label_distribution_parameters = {
    "data_label_class_quantity": {"type": "integer", "min": 1, "default": 2},
    "data_label_distribution_parameter": {"type": "float", "min": 0, "default": 0.5},
}

default_label_distribution = "Uniform"
