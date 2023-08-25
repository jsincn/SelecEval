"""
Contains available feature distributions and their parameters
Also contains the default feature distribution
"""
data_feature_distributions = ["None", "Gaussian"]
data_feature_distribution_parameters = {
    "data_feature_skew_mu": {"type": "float", "default": 0},
    "data_feature_skew_std": {"type": "float", "default": 1},
}

default_feature_distribution = "None"
