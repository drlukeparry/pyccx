PyCCX - Python Library for Calculix
=======================================

.. image:: https://github.com/drlukeparry/pyccx/workflows/Python%20application/badge.svg
    :target: https://github.com/drlukeparry/pyccx/actions
.. image:: https://readthedocs.org/projects/pyccx/badge/?version=latest
    :target: https://pyccx.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
.. image:: https://badge.fury.io/py/PyCCX.svg
    :target: https://badge.fury.io
.. image:: https://badges.gitter.im/pyccx/community.svg
    :target: https://gitter.im/pyccx/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge
    :alt: Chat on Gitter

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
##############

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
* **Materials** (Non-linear Elasto-Plastic Material)

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


Usage
******

The following code excerpt shows an example for creating and running a steady state thermal analysis of model using PyCCX
of an existing mesh generated using the pyccx.mesh.mesher class.

.. code:: python

    from pyccx.core import DOF, ElementSet, NodeSet, SurfaceSet, Simulation
    from pyccx.results import ElementResult, NodalResult, ResultProcessor
    from pyccx.loadcase import  LoadCase, LoadCaseType
    from pyccx.material import ElastoPlasticMaterial

    # Set the path for Calculix in Windows
    Simulation.setCalculixPath('Path')

    # Create a thermal load case and set the timesettings
    thermalLoadCase = LoadCase('Thermal Load Case')

    # Set the loadcase type to thermal - eventually this will be individual analysis classes with defaults
    thermalLoadCase.setLoadCaseType(LoadCaseType.THERMAL)

    # Set the thermal analysis to be a steady state simulation
    thermalLoadCase.isSteadyState = True

    # Attach the nodal and element result options to each loadcase
    # Set the nodal and element variables to record in the results (.frd) file
    nodeThermalPostResult = NodalResult('VolumeNodeSet')
    nodeThermalPostResult.useNodalTemperatures = True

    elThermalPostResult = ElementResult('Volume1')
    elThermalPostResult.useHeatFlux = True

    # Add the result configurations to the loadcase
    thermalLoadCase.resultSet = [nodeThermalPostResult, elThermalPostResult]

    # Set thermal boundary conditions for the loadcase using specific NodeSets
    thermalLoadCase.boundaryConditions.append(
        {'type': 'fixed', 'nodes': 'surface6Nodes', 'dof': [DOF.T], 'value': [60]})

    thermalLoadCase.boundaryConditions.append(
        {'type': 'fixed', 'nodes': 'surface1Nodes', 'dof': [DOF.T], 'value': [20]})

    # Material
    # Add a elastic material and assign it to the volume.
    # Note ensure that the units correctly correspond with the geometry length scales
    steelMat = ElastoPlasticMaterial('Steel')
    steelMat.density = 1.0    # Density
    steelMat.cp =  1.0        # Specific Heat
    steelMat.k = 1.0          # Thermal Conductivity

    analysis.materials.append(steelMat)

    # Assign the material the volume (use the part name set for geometry)
    analysis.materialAssignments = [('PartA', 'Steel')]

    # Set the loadcases used in sequential order
    analysis.loadCases = [thermalLoadCase]

    # Analysis Run #
    # Run the analysis
    analysis.run()

    # Open the results  file ('input') is currently the file that is generated by PyCCX
    results = analysis.results()
    results.load()


The basic usage is split between the meshing facilities provided by GMSH and analysing a problem using the Calculix Solver. Documented
examples are provided in `examples <https://github.com/drlukeparry/pyccx/tree/master/examples>`_ .
