"""
Contains available label distributions and their parameters
Also contains the default label distribution
"""
data_label_distributions = ["Dirichlet", "Uniform", "Discrete"]
data_label_distribution_parameters = {
    "data_label_class_quantity": {"type": "integer", "min": 1, "default": 2},
    "data_label_distribution_parameter": {"type": "float", "min": 0, "default": 0.5},
}

default_label_distribution = "Dirichlet"
