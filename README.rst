PyCCX - Python Calculix
========================
.. image:: https://github.com/drlukeparry/pyccx/workflows/Python%20application/badge.svg
  :target: https://github.com/drlukeparry/pyccx/actions

Provides a library for creating and running 3D FEA simulations using the opensource Calculix FEA Package.

The aims of this project was to provide a simple framework for implemented 3D FEA Analysis using the opensource `Calculix <http://www.calculix.de>`_ solver.
The analysis generation is complimented by use of the relatively recent introduction of the
`GMSH-SDK <http://gmsh.info/>`_ , an extension to GMSH to provide bindings for different programming languages
by the project authors to provide sophisticated 3D FEA mesh generation outside of the GUI implementation . This project aims to provide an integrated simplified method for generating full 3D FEA analysis
for use in research, development and prototyping in a Python environment.

The inception of this project was a result of finding no package available to generate FEA analysis of CAD models in order
to prototype concepts related to 3D printing. The project aims to compliment the work of the `PyCalculix project <https://github.com/spacether/pycalculix>`_, which currently is limited to
providing capabilities to generate 2D Meshes and FEA analysis for 2D planar structures. The potential in the future is to provide
a more generic extensible framework compatible with different opensource and commercial FEA solvers (e.g. Z88, Elmer).
`Learn more <http://lukeparry.uk/>`_.

---------------

If you want to learn more about ``setup.py`` files, check out `this repository <https://github.com/drlukeparry/pyocl/setup.py>`_.


