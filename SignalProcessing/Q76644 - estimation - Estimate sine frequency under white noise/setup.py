# -*- coding: utf-8 -*-
# python setup.py build_ext --inplace
from distutils import _msvccompiler
_msvccompiler.PLAT_TO_VCVARS['win-amd64'] = 'amd64'

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize(Extension("_optimized", ["_optimized.pyx"]),
                          language_level=3),
    include_dirs=[np.get_include()],
)
