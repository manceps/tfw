# -*- coding: utf-8 -*-
import os
from pkg_resources import DistributionNotFound, get_distribution
import glob

try:
    from .constants import SRC_DIR
except: # noqa
    SRC_DIR = 'tfw'

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
    finally:
        del os
        del SRC_DIR

modules = glob.glob(os.path.join(os.path.dirname(__file__), '*.py'))
__all__ = [os.path.basename(f)[:-3] for f in modules if os.path.isfile(f) and not f.endswith('__init__.py')]
from . import *  # noqa
