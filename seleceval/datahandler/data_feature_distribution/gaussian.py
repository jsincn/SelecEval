import torch


class GaussianNoiseTransform(object):
    """
    Add Gaussian noise to a tensor
    """
    def __init__(self, config):
        self.std = config.initial_config['data_config']['data_feature_skew_std']
        self.mean = config.initial_config['data_config']['data_feature_skew_mu']
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)