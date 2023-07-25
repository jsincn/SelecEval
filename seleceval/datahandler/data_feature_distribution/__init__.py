


#data_feature_distribution_dict = {'Uniform': Uniform, 'Normal': Normal}

data_feature_distribution_parameters = \
    {'data_feature_distortion': {'type': 'string', 'allowed': ['none', 'gaussian'], 'default': 'none'},
     'data_feature_distortion_parameter_mu': {'type': 'float', 'min': 0, 'default': 0},
     'data_feature_distortion_parameter_deviation': {'type': 'float', 'min': 0, 'default': 1}}