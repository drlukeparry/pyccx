
# Change Log
All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

### Fixed

### Changed

# [0.2.0] - 2025-08-10

### Added

**General Features:**

- Add support for running PyCCX on Mac OSX platforms - [5900cec2a6004f4ea649c958e82fa0fc1a356b6c](https://github.com/drlukeparry/pyccx/commit/5900cec2a6004f4ea649c958e82fa0fc1a356b6c)

**Analysis Features:**

- Added `pyccx.core.ModelObject` as a base class for all objects in the simulation, which provides generic infrastructure for implementing feature - [8815a89275e252afd75dd76029e98ce58a10e23e](https://github.com/drlukeparry/pyccx/commit/)
- Added `pyccx.core.Meshset` as the base class for all mesh sets in the simulation, which allows for more complex mesh set definitions and operations - [67a229ee8c69f2b02a869c4c1255f62570c662ca](https://github.com/drlukeparry/pyccx/commit/)
- Added `pyccx.core.SurfaceNodeSet` class which is used for defining a flux or distributed load (pressure) BCS within an analysis - [ab264a8a3c9d4c9dcd463c882d0e22d032bf3da9](https://github.com/drlukeparry/pyccx/commit/ab264a8a3c9d4c9dcd463c882d0e22d032bf3da9)
- Added `pyccx.mesh.Ent` enumeration  class used to select and specifying elementary geometric BREP features within GMSH - [89c35fe9eb37d74ba568ec175fbd713c79c60f93](https://github.com/drlukeparry/pyccx/commit/89c35fe9eb37d74ba568ec175fbd713c79c60f93), [31c8896608be06d957149571c7d32b009bd0ecec](https://github.com/drlukeparry/pyccx/commit/31c8896608be06d957149571c7d32b009bd0ecec)
- Added `pyccx.core.Amplitude` object for defining time-dependent BCs - [fadf31cb603bfcb23f06390cbae9bf0bc3b035d0](https://github.com/drlukeparry/pyccx/commit/), [a54e392d56784bce2ca35c522eb703aeeaf1fbde](https://github.com/drlukeparry/pyccx/commit/)
  - Includes definition of target mesh features and time delay - [897d0fcc0c25defcd563fe5192194f95f9ab762a](https://github.com/drlukeparry/pyccx/commit/)
  
**Analysis Features:**

- Simulation can be monitored during the solving phase using the `Simulation.monitor` method, which allows for real-time updates of the simulation status and results - [df0e83c3472c43afd292c49a66687615538703f2](https://github.com/drlukeparry/pyccx/commit/df0e83c3472c43afd292c49a66687615538703f2)
- Calculix output stream is piped and data is extracted to obtain runtime information
- Total elapsed runtime is calculated and available in the `pyccx.analysis.Simulation` class - [77766655f61fedd5cbed065f63276b4aa7314da2](https://github.com/drlukeparry/pyccx/commit/)
- Checks added for verifying calculix executable path and compiled version - [15e25fc5b19bbb5850226bb3ea581f6ff51cd19c](https://github.com/drlukeparry/pyccx/commit/)

**Load Case Features:**

- Added several features in the `LoadCase` class to allow for more complex simulations scenarios, including 
  - Steady State Analysis - [14e91a14c8c03e7e54e4083c3e36e736b02c6550](https://github.com/drlukeparry/pyccx/commit/)
  - Default, initial, and minimum and maximum time-stepping options 
  - Total simulation duration
  - Enable automatic incrementation
  - Enable Nonlinear analysis [24f3d7918a2df6a37c09fa63650dc24d580f129b](https://github.com/drlukeparry/pyccx/commit/)
- Added a method to reset boundary condition with sequential application of loadcases [13cbbee03d0c05d31ade55621d7aad34733e2c27](https://github.com/drlukeparry/pyccx/commit/)

**Postprocessing Features:**

This releases includes significant improvements to the postprocessing features of PyCCX, including:
- Refactored `Result` class to allow for more complex result extraction and storage
  - Added `ResultsValue` class which is used to specify requested simulation output features - [88c29ae537074a78cd28bf3c79a5f71aca171097](https://github.com/drlukeparry/pyccx/commit/)
  - Added a method to calculate the 2D and 3D von Mises stress from the cauchy stresses - [d9c40817acf5777701a30584ec6225434b7f32cb](https://github.com/drlukeparry/pyccx/commit/)
  - Drastically improved parsing of calculix output files (.frd, .dat) [b9a8c88dc73c55126c007d35b0f18a9db155cf87](https://github.com/drlukeparry/pyccx/commit/)
  - `ResultProcessor` extract nodal and elements results data from the output files, which was previously unavailable [dcb19e965a650a37241cb4ca372e5842ed461556](https://github.com/drlukeparry/pyccx/commit/)
  - Correctly stores the output generated during the solving phase of the Calculix simulation - [6c8d341e7cced4d99cdfa4cdd1f5522f51e6e472](https://github.com/drlukeparry/pyccx/commit/)
  - Added option for Calculiux to not expand shell elements when requested `NodalResults.expandShellElements` - [f52309b907dfdd4484a2b32e2d61ef3cc11225d2](https://github.com/drlukeparry/pyccx/commit/)
  - Added method `ResultProcessor.clearResults` - [ed15b15d8fcbe7d422062e573a5e5a375e0462d6](https://github.com/drlukeparry/pyccx/commit/)
- Added VTK Exporter for exporting results to VTK format, which can be used for visualisation in ParaView or other VTK-compatible software - [6c841317d977cace690a2c3b984b61686a5e4371](https://github.com/drlukeparry/pyccx/commit/6c841317d977cace690a2c3b984b61686a5e4371)
  - Added `utils.exportToVTK` method that is used to export the results to native VTK format
  - Added `utils.exportToPVD` method to export all timesteps to a _.PVD_ format - [0b944486591ba2dfe56efb7d9e62d7bdfea1cddd](https://github.com/drlukeparry/pyccx/commit/0b944486591ba2dfe56efb7d9e62d7bdfea1cddd)

**Meshing Features:**
- Added the method `Mesher.setRecombineSurfaces` and `Mesher.recombineMesh` offering functionality to recombine surface meshes and corresponding 3D volumes
  - Uses internal GMSH method for recombining surface meshes to 3D volumes - [5f2564430c51fa6aba746c7f4cbf8d575ce5bb5c](https://github.com/drlukeparry/pyccx/commit/5f2564430c51fa6aba746c7f4cbf8d575ce5bb5c)
- Added `pyccx.mesh.RecombinationAlgorithm` enumeration class which is used to specify the recombination algorithm used in GMSH - [7b433b75c314313752da7e7578d43849246b37fa](https://github.com/drlukeparry/pyccx/commit/7b433b75c314313752da7e7578d43849246b37fa)
  - Usage is specified in `Mesher.recombinationAlgorithm`
- Added a several element families and types for matching between GMSH and Calculix Native Types including:  [14e91a14c8c03e7e54e4083c3e36e736b02c6550](https://github.com/drlukeparry/pyccx/commit/14e91a14c8c03e7e54e4083c3e36e736b02c6550)
    - `pyccx.mesh.elements.ElementType` enumeration class which is used to specify the element type (e.g. planar, shell, 3D continuum, etc.)
    - `pyccx.mesh.elements.ElementFamilies` class enumeration which is used to specify the base geometric element family used in the simulation
    - Element mappings between GMSH and Calculix native types - including face orders
    - Mask for specifying the mapping between element face and nodal orders
- Added GMSH meshing algorithm enumeration class which is used to specify the meshing algorithm used in GMSH - [2f4417b095a165aa48846d826612f92f2a94b671](https://github.com/drlukeparry/pyccx/commit/2f4417b095a165aa48846d826612f92f2a94b671)
  - Includes GMSH 2D Meshing algorithm options (`pyccx.mesh.MeshingAlgorithm2D`) and 3D Mesh algorithms options (`pyccx.mesh.MeshingAlgorithm3D`) 
- Add Mesh element assignments to ensure different element types are correctly mapped between GMSH and Calculix - [7cbaebed880c75b1a85d1376fb66466649155ccf](https://github.com/drlukeparry/pyccx/commit/7cbaebed880c75b1a85d1376fb66466649155ccf)
  - Specified assignments are stored in a `Mesh.meshAssignments` dictionary
- Added method `Mesh.identifyUnassignedElements` to the `Mesh` class which identifies elements that are not assigned to a specific element type - [c52cbb11bad0af4116462b72bc66b555984ccebd](https://github.com/drlukeparry/pyccx/commit/c52cbb11bad0af4116462b72bc66b555984ccebd)
- Added method `Mesher.open` to load an existing GMSH model file into the `Mesher` class - [da42c26fdc94f0cf8d65be98be3c9674ed47ce22](https://github.com/drlukeparry/pyccx/commit/da42c26fdc94f0cf8d65be98be3c9674ed47ce22)
- Added method `Mesher.getAllPhysicalGroupElements` to the `Mesher` class which returns all physical group elements in the mesh - [a0cd63258e9e9f8ce4f53797d2a42cd3cf3ad7af](https://github.com/drlukeparry/pyccx/commit/a0cd63258e9e9f8ce4f53797d2a42cd3cf3ad7af)
- Added method `Mesher.clearPhyiscalGroups` to remove all physical groups within the Mesh model - [ee712094168641e7c71d7f17945645f363826edf](https://github.com/drlukeparry/pyccx/commit/ee712094168641e7c71d7f17945645f363826edf)
- Added method `Mesher.getElementType` and update methods to correctly generate the mesh [c22d94a1651d4609384db1484c59a789f0440db5](https://github.com/drlukeparry/pyccx/commit/c22d94a1651d4609384db1484c59a789f0440db5)
- Added method `Mesher.getFacesFromId` to obtain the current correctly orientated face lists for a given surface id - [df0e83c3472c43afd292c49a66687615538703f2](https://github.com/drlukeparry/pyccx/commit/df0e83c3472c43afd292c49a66687615538703f2)
- Mesher now directly exports the mesh to a Calculix input file using the `Mesher.writeMesh` method - [f57dcb9a7c7f23e4fd3022c91a8edaa0b8b1de0b](https://github.com/drlukeparry/pyccx/commit/f57dcb9a7c7f23e4fd3022c91a8edaa0b8b1de0b)
  - Resolves various issues with exporting from the GMSH .msh native format, and instead directly produces the correct format for Calculix

### Fixed 

- Update NodeSet and ElementSet to correctly export the node and element ids - [fe3e6b70fda75793cd9e70b19b9a3b5797d95204](https://github.com/drlukeparry/pyccx/commit/fe3e6b70fda75793cd9e70b19b9a3b5797d95204)
- Fix maximum number of element ids per line for an ElementSet - [1cdb3d174d1d0df5366b59efa94defaf8bac3066](https://github.com/drlukeparry/pyccx/commit/1cdb3d174d1d0df5366b59efa94defaf8bac3066)
- Correct exporting of decimal numbers to 5 decimal places in the input file - [c0c7f19b0d564b730fec520935123e8ec459113a](https://github.com/drlukeparry/pyccx/commit/c0c7f19b0d564b730fec520935123e8ec459113a)
- Elements sets are collected for the material assignments - [da81bb96cd1d1ebb3d4a45a5053d8b9a807c4436](https://github.com/drlukeparry/pyccx/commit/da81bb96cd1d1ebb3d4a45a5053d8b9a807c4436)
- Bug Fix: Surface Sets are correctly exported in the Simulation class - [d4d4ff806dd9fbc18b786e31d4aa5b95c82ff66f](https://github.com/drlukeparry/pyccx/commit/d4d4ff806dd9fbc18b786e31d4aa5b95c82ff66f)
- Bug Fix: Force components are correctly written to the input file - [3705feb7ac9b11340e5b444eddaf2195feff561a](https://github.com/drlukeparry/pyccx/commit/3705feb7ac9b11340e5b444eddaf2195feff561a)
- Bug Fix: General improvements to consistency for identifying nodes within a GMSH model in the `pyccx.mesh.Mesher` class - [deaa0f286849babdf32e1a0a07c8a3cd2871a1fe](https://github.com/drlukeparry/pyccx/commit/deaa0f286849babdf32e1a0a07c8a3cd2871a1fe)
- General updates to catch issues and raise program exceptions [9d436f3d56a1835a0fe3b879a21092a70f18b389](https://github.com/drlukeparry/pyccx/commit/9d436f3d56a1835a0fe3b879a21092a70f18b389), [c3f4e305413d6f0c9f8c129d0c6806c27def92c6](https://github.com/drlukeparry/pyccx/commit/c3f4e305413d6f0c9f8c129d0c6806c27def92c6)
- `NodalResult` and `ElementResult` attributes renamed [b8d7eba9050b436afe442c61a0c599b282556855](https://github.com/drlukeparry/pyccx/commit/b8d7eba9050b436afe442c61a0c599b282556855), [bf06c16242523853ab69ab0bd59aecd9cc838788](https://github.com/drlukeparry/pyccx/commit/bf06c16242523853ab69ab0bd59aecd9cc838788)
- Use the current 'scratch' directory to loading thesSimulation's result files - [aa1976d9a01578ae1958f38fcccc382271b18b0b](https://github.com/drlukeparry/pyccx/commit/aa1976d9a01578ae1958f38fcccc382271b18b0b)

### Changed

- Direct path for Calculix solver executable on Window is now used for consistency - [b48144e7407c1ff20bb43e6e9c9b1ed7e51b7023](https://github.com/drlukeparry/pyccx/commit/b48144e7407c1ff20bb43e6e9c9b1ed7e51b7023)
- Updated build system to use hatchling for building the PyCCX package - [e27680ff587f53211bb8646d8194415bfe1dd41f](https://github.com/drlukeparry/pyccx/commit/e27680ff587f53211bb8646d8194415bfe1dd41f)
  - Migrated to `pyproject.toml` for python package metadata and build configuration
- Ruff linting now used for code formatting and linting - [1aa76e4c5b4d4195574cd84cde42be3cebb30fbb](https://github.com/drlukeparry/pyccx/commit/1aa76e4c5b4d4195574cd84cde42be3cebb30fbb)
- Fixed typo for emissivity attribute for `pyccx.bc.Radiation` BC - [5577091a97ae74689fb3aaafbf83de135cb174bf](https://github.com/drlukeparry/pyccx/commit/5577091a97ae74689fb3aaafbf83de135cb174bf)
- Updated the `pyccx` module imports - [33fe5a9939b5cc2cfea7ab81d47602b194009e99](https://github.com/drlukeparry/pyccx/commit/33fe5a9939b5cc2cfea7ab81d47602b194009e99)
- Updated project inline documentation - [03743c3ae2dfe2540de034b49f771f2235c1c417](https://github.com/drlukeparry/pyccx/commit/03743c3ae2dfe2540de034b49f771f2235c1c417), [3e12ceefe2c65d67308a2e4d05a2ccdd428a7705](https://github.com/drlukeparry/pyccx/commit/3e12ceefe2c65d67308a2e4d05a2ccdd428a7705), [d1cd854ff629fef6cae8e9651a0acf470deca273](https://github.com/drlukeparry/pyccx/commit/d1cd854ff629fef6cae8e9651a0acf470deca273)
- Updated `pyccx.loadcase.LoadCaseType` to be derived from an `IntEnum` for unique IDs [156257dc65d1a2820ba5477ea428468297421adc](https://github.com/drlukeparry/pyccx/commit/156257dc65d1a2820ba5477ea428468297421adc)
- Minimum version of Python is now 3.9, because 3.8 is now considered unmaintained
- Minimum version of GMSH requires is 4.14 [6156c8339731a1f5359b9396e97a9707eb101320](https://github.com/drlukeparry/pyccx/commit/6156c8339731a1f5359b9396e97a9707eb101320)

## [0.1.2] - 2023-07-11

  The first release of PyCCX on PyPI.

