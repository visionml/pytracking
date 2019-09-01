from pytracking import TensorList
import random


class TrackerParams:
    """Class for tracker parameters."""
    def set_default_values(self, default_vals: dict):
        for name, val in default_vals.items():
            if not hasattr(self, name):
                setattr(self, name, val)


class FeatureParams:
    """Class for feature specific parameters"""
    def __init__(self, *args, **kwargs):
        if len(args) > 0:
            raise ValueError

        for name, val in kwargs.items():
            if isinstance(val, list):
                setattr(self, name, TensorList(val))
            else:
                setattr(self, name, val)


def Choice(*args):
    """Can be used to sample random parameter values."""
    return random.choice(args)
