"""
Package containing the different client filters applicable before client selection
"""

from .performance_based_filter import PerformanceBasedFilter

__all__ = ["PerformanceBasedFilter", "filter_dict"]

filter_dict = {
    "performance_based": PerformanceBasedFilter
    }