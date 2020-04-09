# -*- coding: utf-8 -*-

# Learn more: https://github.com/lukeparry/pyccx/setup.py

from setuptools import setup, find_packages

# minimal requirements for installing pyccx
# note that `pip` requires setuptools itself
requirements_default = set([
    'numpy',     # all data structures
    'gmsh-sdk',  # Required for meshing geometry
    'setuptools'  # used for packaging
])

# "easy" requirements should install without compiling
# anything on Windows, Linux, and Mac, for Python 2.7-3.4+
requirements_easy = set([
    'setuptools',  # do setuptools stuff
    'colorlog'])   # log in pretty colors


# requirements for building documentation
requirements_docs = set([
    'sphinx',
    'jupyter',
    'numpy',
    'pypandoc',
    'autodocsumm'
    'sphinx_rtd_theme'])

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='PyCCX',
    version='0.1.0',
    description='Simulation FEA enviornment for Python built upon Calculix and GMSH',
    long_description=readme,
    author='Luke Parry',
    author_email='dev@lukeparry.uk',
    url='https://github.com/drlukeparry/pyccx',
    keywords='FEA, Finite Element Analysis, Simulation, Calculix, GMSH',
    classifiers=[
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Natural Language :: English',
        'Topic :: Scientific/Engineering'],
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=list(requirements_default),
)

