PyCCX - Python Library for Calculix
=======================================

.. image:: https://github.com/drlukeparry/pyccx/workflows/Python%20application/badge.svg
    :target: https://github.com/drlukeparry/pyccx/actions
.. image:: https://readthedocs.org/projects/pyccx/badge/?version=latest
    :target: https://pyccx.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
.. image:: https://badge.fury.io/py/PyCCX.svg
    :target: https://badge.fury.io
.. image:: https://img.shields.io/badge/Made%20with-Python-1f425f.svg
   :target: https://www.python.org/
.. image:: https://img.shields.io/pypi/l/pyccx.svg
   :target: https://pypi.python.org/pypi/pyccx/
..  image:: https://img.shields.io/pypi/pyversions/pyccx.svg
   :target: https://pypi.python.org/pypi/pyccx/

PyCCX - a library for creating and running 3D FEA simulations using the opensource Calculix FEA Package.

The aim of this project was to provide a framework for implemented 3D FEA Analysis using the opensource
`Calculix <http://www.calculix.de>`_ solver. The analysis is complimented by use of the recent introduction of the
`GMSH-SDK <http://https://gitlab.onelab.info/gmsh/gmsh/api>`_ , an extension to `GMSH <http://gmsh.info/>`_ to provide
API bindings for different programming languages by the project authors to provide sophisticated 3D FEA mesh
generation outside of the GUI implementation.

This project aims to provide an integrated approach for generating full
2D and 3D structural and thermal FEA analysis for use in research, development and prototyping all inside a
Python environment. The functionality targets the delivery of automated scripted approaches for performing FEA simulations,
in particular for use assessing the sensitivity of design and material inputs on the response of a system structure, that
can be used as part of parametric optimisation studies.

This intends to remove requirement to setup each analysis manually using a GUI such as prepromax or GMSH.

Along with setting up and processing the analysis, numerous convenience functions are included to consistently interface
between both the Calculix and GMSH functionality within a single python environment.

Structure
##############

PyCCX framework consists of classes for specifying common components on the pre-processing stage, including the following
common FE workflow for performing a simulation:

* Generation of both 2D and 3D compatible analysis meshes for use with Calculix via GMSH
* Creation and assignment of thermal and mechanical boundary conditions for use in analyses
* Creation of multiple time (in)-dependent load cases
* Creation and assignment of multiple material models and element types through a single analysis
* Control and monitoring the Calculix simulation execution
* Processing and extraction of results obtained from Calculix

A meshing infrastructure provides an interface with GMSH for performing the meshing routines and for associating
physical boundary conditions with the elements/faces generated from geometrical entities obtained from CAD models,
typically by importing .step files.

The simulation class assembles the mesh and corresponding mesh identifier sets (Element, Nodal and Surfaces)
in conjunction with the applied boundary conditions for each specified load-case within an analysis. The analysis
is then exported as a Calculix input deck, and then performs the execution to the Calculix solver. The simulation
can be additionally monitored within the Python environment.

The results obtained upon completion of the analysis can be processes, to extract individual nodal and elemental quantities
predicted in the analysis output. The results can also be exported to an unstructured VTK format for visualisation in
Paraview.

Currently the analysis is unit-less, therefore the user should ensure that all constant, material parameters, boundary
conditions, and geometric lengths are consistent - by default GMSH assumes 'mm' units when importing BRep CAD models.

Current Features
******************

Meshing:
---------
Meshing is performed using the GMSH-SDK, which provides a Python interface to the GMSH meshing library. The features
within pyccx provided higher-level functionality building across existing GMSH functionality. The library mainly
facilitates setting up the analysis consistently within a single environment, such as mapping geometrical FE elements
into compatible Calculix types with consistent nodal ordering. Additional features available for meshing include:

* Integration with GMSH for generation 3D FEA Meshes
* Cleaning and merging of CAD assemblies using internal functionality provided by GMSH
* Creation and assignment of NodeSet, ElementSet, SurfaceSet from mesh features applied for boundary conditions
* Attachment of boundary conditions to geometrical CAD entities via GMSH (native .step import supported via OCC)

FEA Capabilities:
-------------------

* **Boundary Conditions**: (Acceleration, Convection, Fixed Displacements, Forces, Fluxes, Pressure, Radiation)
* **Loadcase Types** (Structural Static, Thermal, Coupled Thermo-Mechanical)
* **Materials** (Non-linear Elasto-Plastic Material) with user defined stress-strain curves and physical properties
* **Results** (Selection of exported results for nodal and element data per loadcase)
* **Analysis Types** configurable solver control (:auto-incrementing timestep, non-linear analysis)

Results Processing:
----------------------
* Element and Nodal Results can be obtained across each timesteps
* Results can be processed and visualised using the `pyccx.results` module
* Extraction of node and element results directly from the Calculix .frd and datafile
* Export of results to VTK file format for visualisation directly in Paraview


Installation
*************
PyCCX is multi-platform as a source based pythonpackage. This can be installed along with dependencies for GMSH automatically
using the following commands:

.. code:: bash

    pip install gmsh
    pip install pyccx

alternatively, the package can be installed using the uv library:

.. code:: bash

    uv pip install gmsh
    uv pip install pyccx

Calculix Solver
*****************

Depending on your environment, you will need to install the latest version of Calculix. This can be done through
conda-forge `calculix package <https://anaconda.org/conda-forge/calculix>`_ in the Anaconda distribution,

.. code:: bash

    conda install -c conda-forge calculix

However, it is suggested that the most reliable mode is downloading the latest distribution of Calculix directly.

**Windows:**

The solver be separately obtained from within the distribution of `prepromax <https://prepomax.fs.um.si>`_

**Linux:**

The latest version of Calculix can be installed from the packages available within your linux distribution

**Mac OS X:**

Calculix can be installed using the `Homebrew <https://brew.sh/>`_ package manager. This requires the appropriate XCode
compiler environment to be installed. Once this is done, Calculix can be installed using the following command:

.. code:: bash

    brew tap costerwi/homebrew-calculix
    brew install calculix-ccx

The path of the installed Calculix solver executable should be obtained, which is dependent on the configuration of the
brew installation.

Usage
*************

The Calculix solver executable needs to be available in the system path, or the path to the executable needs to be manually
specified. Across all platforms the direct path of the calculix solver executable needs to be initialised before any
further use.

.. code:: python

    from pyccx.core import Simulation

    # Set the path for Calculix in Windows
    Simulation.setCalculixPath('Path')


The following code excerpt shows part of an example for creating and running a steady state thermal analysis of model
using PyCCX of an existing mesh generated using the `pyccx.mesh.mesher` class.

.. code:: python

    from pyccx.core import DOF, ElementSet, NodeSet, SurfaceSet, Simulation
    from pyccx.results import ElementResult, NodalResult, ResultProcessor
    from pyccx.loadcase import  LoadCase, LoadCaseType
    from pyccx.material import ElastoPlasticMaterial

    # Set the path for Calculix in Windows
    Simulation.setCalculixPath('Path')

    # Create a Simulation object based on the supplied mesh model (defined separately)
    analysis = Simulation(myMeshModel)

    # Optionally set the working the base working directory
    analysis.setWorkingDirectory('.')


    # Create an ElementSet  and NodeSet for the entire volume of named model ('PartA')
    myMeshModel.setEntityName((Ent.Volume, 1), 'PartA') # Set the name of the GMSH volume to 'PartA'
    volElSet = ElementSet('volElSet', myMeshModel.getElementIds((Ent.Volume,1)))
    volNodeSet = NodeSet('VolumeNodeSet', myMeshModel.getNodesFromVolumeByName('PartA'))

    analysis.initialConditions.append({'type': 'temperature', 'set': 'VolumeNodeSet', 'value': 0.0})

    # Create a thermal load case and set the timesettings
    thermalLoadCase = LoadCase('Thermal_Load_Case')

    # Set the loadcase type to thermal - eventually this will be individual analysis classes with defaults
    thermalLoadCase.setLoadCaseType(LoadCaseType.THERMAL)

    # Set the thermal analysis to be a steady state simulation
    thermalLoadCase.isSteadyState = True
    thermalLoadCase.setTimeStep(0.5, 0.5, 5.0)

    # Attach the nodal and element result options to each loadcase
    # Set the nodal and element variables to record in the results (.frd) file
    nodeThermalPostResult = NodalResult('volNodeSet')
    nodeThermalPostResult.temperature = True

    elThermalPostResult = ElementResult('Volume1')
    elThermalPostResult.heatFlux = True

    # Add the result configurations to the loadcase
    thermalLoadCase.resultSet = [nodeThermalPostResult, elThermalPostResult]

    # Set thermal boundary conditions for the loadcase using specific NodeSets
    thermalLoadCase.boundaryConditions.append(
        {'type': 'fixed', 'nodes': 'surfaceNodesA', 'dof': [DOF.T], 'value': [60]})

    thermalLoadCase.boundaryConditions.append(
        {'type': 'fixed', 'nodes': 'surfaceNodesB', 'dof': [DOF.T], 'value': [20]})

    # Material
    # Add a elastic material and assign it to the volume.
    # Note ensure that the units correctly correspond with the geometry length scales
    steelMat = ElastoPlasticMaterial('Steel')
    steelMat.E = 210000.      # [MPa] Young's Modulus
    steelMat.alpha_CTE = [25e-6, 23e-6, 24e-6]  # Thermal Expansion Coefficient
    steelMat.density = 1.0    # Density
    steelMat.cp =  1.0        # Specific Heat
    steelMat.k = 1.0          # Thermal Conductivity


    # The material and material type is assigned to the elements across the part
    analysis.materialAssignments = [
        SolidMaterialAssignment("solid_material", elementSet=volElSet, material=steelMat)
    ]

    # Set the loadcases used in sequential order
    analysis.loadCases = [thermalLoadCase]

    # Run the analysis #
    analysis.run()

    # Open the results  file ('input') is currently the file that is generated by PyCCX
    results = analysis.results()
    results.load()

    # Export the results to VTK format as a significant timestep for post-processing
    import pyccx.utils.exporters as exporters
    exporters.exportToVTK('result.vtu', results, inc=-1)


The basic usage is split between the meshing facilities provided by GMSH and analysing a problem using the Calculix
Solver. Further documented examples are provided in `examples <https://github.com/drlukeparry/pyccx/tree/master/examples>`_ .

The current changelog is found in the `CHANGELOG <https://github.com/drlukeparry/pyccx/tree/dev/CHANGELOG.md'>`_ .