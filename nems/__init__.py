"""
NEMS
====

Neural Encoding Models

A python package for fitting and visualizing neural encoding
(stimulus-response) models.

Modules
-------
models
objectives
tentbasis
utilities
visualization
metrics
nonlinearities

"""

__author__ = 'nirum'
__version__ = '0.2.1'
__all__ = ['models', 'objectives', 'tentbasis', 'utilities', 'visualization']

from .models import *
from . import tentbasis
from . import objectives
from . import visualization as viz
