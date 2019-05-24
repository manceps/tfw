# -*- coding: utf-8 -*-
import os
from pkg_resources import DistributionNotFound, get_distribution
import glob

try:
    from .constants import SRC_DIR
except: # noqa
    SRC_DIR = os.path.dirname(os.path.abspath(__file__))


try:
    dist_name = __name__
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    __version__ = 'unknown'
finally:
    del get_distribution, DistributionNotFound
    del dist_name

if __version__ == 'unknown':
    try:
        __version__ = open(os.path.join(SRC_DIR, 'VERSION')).read().strip()
    except:  # noqa
        pass


modules = list(glob.glob(os.path.join(SRC_DIR, '*.py')))
print(modules)
__all__ = [os.path.basename(f)[:-3] for f in modules if not f.endswith('__init__.py')]

print(__all__)

from . import *  # noqa
del os, glob
del SRC_DIR, modules
