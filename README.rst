PyCCX - Python Calculix
========================

.. image:: https://badges.gitter.im/pyccx/community.svg
   :alt: Join the chat at https://gitter.im/pyccx/community
   :target: https://gitter.im/pyccx/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge
.. image:: https://github.com/drlukeparry/pyccx/workflows/Python%20application/badge.svg
    :target: https://github.com/drlukeparry/pyccx/actions
.. image:: https://readthedocs.org/projects/pyccx/badge/?version=latest
    :target: https://pyccx.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

Provides a library for creating and running 3D FEA simulations using the opensource Calculix FEA Package.

The aims of this project was to provide a simple framework for implemented 3D FEA Analysis using the opensource `Calculix <http://www.calculix.de>`_ solver.
The analysis is complimented by use of the recent introduction of the
`GMSH-SDK <http://https://gitlab.onelab.info/gmsh/gmsh/api>`_ , an extension to `GMSH <http://gmsh.info/>`_  to provide API bindings for different programming languages
by the project authors to provide sophisticated 3D FEA mesh generation outside of the GUI implementation. This project aims to provide an integrated approach for generating full 3D FEA analysis
for use in research, development and prototyping in a Python environment. Along with setting up and processing the analysis,
convenience functions are included.

The inception of this project was a result of finding no native Python/Matlab package available to perfom full non-linear FEA analysis
of 3D CAD models in order to prototype a concept related to 3D printing. The project aims to compliment the work of
the `PyCalculix project <https://github.com/spacether/pycalculix>`_, which currently is limited to providing capabilities
to generate 2D Meshes and FEA analysis for 2D planar structures. The potential in the future is to provide
a more generic extensible framework compatible with different opensource and commercial FEA solvers (e.g. Abaqus, Marc, Z88, Elmer).

An interface that built upon GMSH was required to avoid the use of the GUI, and the domain specific .geo scripts.
`Learn more <http://lukeparry.uk/>`_.

Structure
###########

PyCCX framework consists of classes for specifying common components on the pre-processing phase, including the following
common operations:

* Mesh generation
* Creating and applying boundary conditions
* Creating load cases
* Creating and assigning material models
* Performing the simulation

In addition, a meshing class provides an interface with GMSH for performing the meshing routines and for associating
boundary conditions with the elements/faces generated from geometrical CAD entities. The Simulation class assembles the
analysis and performs the execution to the Calculix Solver. Results obtained upon completion of the analysis can be processed.
Currently the analysis is unit-less, therefore the user should ensure that all constant, material paramters, and geometric
lengths are consistent - by default GMSH assumes 'mm' units.

Current Features
******************

**Meshing:**

* Integration with GMSH for generation 3D FEA Meshes (Tet4, Tet10 currently supported)
* Merging CAD assemblies using GMSH
* Attaching boundary conditions to Geometrical CAD entities

**FEA Capabilities:**

* **Boundary Conditions** (Acceleration, Convection, Fixed Displacements, Forces, Fluxes, Pressure, Radiation)
* **Loadcase Types** (Structural Static, Thermal, Coupled Thermo-Mechanical)
* **Materials** (Non-linear Elastic)

**Results Processing:**

* Element and Nodal Results can be obtained across timesteps


Installation
*************
Installation is currently supported on Windows, all this further support will be added for
Linux environments. PyCCX can be installed along with dependencies for GMSH automatically using.

.. code:: bash

    pip install pyccx


Depending on your environment, you will need to install the latest version of Calculix. This can be done through
the conda-forge `calculix package <https://anaconda.org/conda-forge/calculix>`_ in the Anaconda distribution,

.. code:: bash

    conda install -c conda-forge calculix


or alternatively downloading the package directly. On Windows platforms the path of the executable needs to be initialised before use.

.. code:: python

    from pyccx.core import Simulation

    # Set the path for Calculix in Windows
    Simulation.setCalculixPath('Path')


USAGE
******

The basic usage is split between the meshing facilities provided by GMSH and analysing a problem using the Calculix Solver. Documented
examples are provided in `examples <https://github.com/drlukeparry/pyccx/tree/master/examples>`_ .