import os
from setuptools import setup, find_packages

# load __version__ without importing anything
version_file = os.path.join(
    os.path.dirname(__file__),
    'pyccx/version.py')

with open(version_file, 'r') as f:
    # use eval to get a clean string of version from file
    __version__ = eval(f.read().strip().split('=')[-1])

# load README.md as long_description
long_description = ''
if os.path.exists('README.rst'):
    with open('README.rst', 'r') as f:
        long_description = f.read()

# minimal requirements for installing pyccx
# note that `pip` requires setuptools itself
requirements_default = set([
    'numpy',      # all data structures
    'gmsh',       # Required for meshing geometry
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
    'sphinx_rtd_theme',
    'pypandoc',
    'autodocsumm'])

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='PyCCX',
    version=__version__,
    description='Simulation and FEA environment for Python built upon Calculix and GMSH',
    long_description=long_description,
    long_description_content_type = 'text/x-rst',
    author='Luke Parry',
    author_email='dev@lukeparry.uk',
    url='https://github.com/drlukeparry/pyccx',
    keywords='FEA, Finite Element Analysis, Simulation, Calculix, GMSH',
    python_requires='>=3.5',
    classifiers=[
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Natural Language :: English',
        'Topic :: Scientific/Engineering'],
    license="",
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=list(requirements_default),
    extras_require={'easy': list(requirements_easy),
                    'docs': list(requirements_docs)},

    project_urls={
        'Documentation': 'https://pyccx.readthedocs.io/en/latest/',
        'Source': 'https://github.com/drylukeparry/pyccx/pyccx/',
        'Tracker': 'https://github.com/drlukeparry/pyccx/issues'
    }


)


