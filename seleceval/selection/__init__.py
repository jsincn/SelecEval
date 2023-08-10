"""
Client Selection algorithms for federated learning.
"""
from .active import ActiveFL
from .cep import CEP
from .client_selection import ClientSelection
from .fedcs import FedCS
from .powd import PowD
from .random_selection import RandomSelection

__all__ = ['ActiveFL', 'CEP', 'FedCS', 'PowD', 'RandomSelection', 'ClientSelection', 'algorithm_dict']

algorithm_dict = {'FedCS': FedCS, 'PowD': PowD, 'random': RandomSelection, 'CEP': CEP, 'ActiveFL': ActiveFL}