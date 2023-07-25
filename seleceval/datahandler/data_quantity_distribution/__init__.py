from .dirichlet import Dirichlet
from .uniform import Uniform

__all__ = ['Dirichlet', 'Uniform', 'data_quantity_distribution_dict', 'data_quantity_distribution_parameters', 'default_quantity_distribution']

data_quantity_distribution_dict = {'Uniform': Uniform, 'Dirichlet': Dirichlet}

data_quantity_distribution_parameters = \
    {'data_quantity_base_parameter': {'type': 'float', 'min': 0.001, 'max': 1, 'default': 0.01},
     'data_quantity_skew_parameter': {'type': 'float', 'min': 0, 'default': 0.5},
     'data_quantity_min_parameter': {'type': 'integer', 'min': 0, 'default': 10}}

default_quantity_distribution = 'Uniform'