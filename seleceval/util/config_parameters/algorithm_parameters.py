algorithm_parameter_dict = {
    'FedCS':
        {'type': 'dict', 'default': {},
         'schema': {
             'c': {'type': 'float', 'min': 0, 'max': 1, 'default': 0.2},
             'fixed_client_no': {'type': 'boolean', 'default': True},
             'pre_sampling': {'type': 'float', 'min': 0, 'max': 1, 'default': 0.4}
         }
         },
    'PowD':
        {'type': 'dict', 'default': {},
         'schema': {
             'c': {'type': 'float', 'min': 0, 'max': 1, 'default': 0.2},
             'pre_sampling': {'type': 'float', 'min': 0, 'max': 1, 'default': 0.4}
         }
         },
    'CEP':
        {'type': 'dict', 'default': {},
         'schema': {
             'c': {'type': 'float', 'min': 0, 'max': 1, 'default': 0.2}
         }
         },
    'ActiveFL':
        {'type': 'dict', 'default': {},
         'schema': {
             'alpha1': {'type': 'float', 'min': 0, 'max': 1, 'default': 0.75},
             'alpha2': {'type': 'float', 'min': 0, 'max': 1, 'default': 0.01},
             'alpha3': {'type': 'float', 'min': 0, 'max': 1, 'default': 0.1},
             'c': {'type': 'float', 'min': 0, 'max': 1, 'default': 0.2}
         }
         },
    'random':
        {'type': 'dict', 'default': {},
         'schema': {
             'c': {'type': 'float', 'min': 0, 'max': 1, 'default': 0.2}
         }
         }
}
