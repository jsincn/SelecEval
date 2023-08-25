"""
This module contains methods of skewing data features
"""
from .gaussian import GaussianNoiseTransform

#data_feature_distribution_dict = {'Uniform': Uniform, 'Normal': Normal}
data_feature_distribution_dict = {'None': None, 'Gaussian': GaussianNoiseTransform}