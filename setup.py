#!/usr/bin/env python

from distutils.core import setup, Extension
import Cython.Distutils
import numpy

setup(name             = "numpy_c_ext_example",
      version          = "1.0",
      description      = "Example code for blog post.",
      author           = "J. David Lee",
      author_email     = "contact@crumpington.com",
      maintainer       = "contact@crumpington.com",
      url              = "https://www.crumpington.com",
      cmdclass={'build_ext': Cython.Distutils.build_ext},
      ext_modules      = [
          Extension(
              'lib.simple1', ['src/simple1.c'],
              include_dirs=[numpy.get_include()],
              extra_compile_args=["-Ofast", "-march=native"]),
          Extension(
              'lib.simple2', ['src/simple2.c'],
              include_dirs=[numpy.get_include()],
              extra_compile_args=["-Ofast", "-march=native"]),
          Extension('lib.sim2', ['lib/sim2.pyx'])
      ], 
      
)
