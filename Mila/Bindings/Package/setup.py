"""
Marks the distribution as binary.

Everything else is declarative in pyproject.toml. This file exists for one reason:
the compiled extension is staged into src/mila by the CMake build rather than built
by setuptools, so setuptools sees no ext_modules and would tag the result
py3-none-any -- a pure-Python wheel that pip would happily install under the wrong
interpreter and platform, where the extension cannot load. Declaring
has_ext_modules() forces the real cp3XX-cp3XX-<platform> tag.
"""

from setuptools import setup
from setuptools.dist import Distribution


class BinaryDistribution(Distribution):
    def has_ext_modules(self) -> bool:
        return True


setup(distclass=BinaryDistribution)
