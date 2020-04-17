# -*- coding: utf-8 -*-
"""
@author: Luke Parry
nod"""

import pyccx

from pyccx.mesh import ElementType, Mesher

from pyccx.core import ElementSet, NodeSet, SurfaceSet, Simulation
from pyccx.results import ElementResult, NodalResult, ResultProcessor
from pyccx.loadcase import  LoadCase, LoadCaseType
from pyccx.material import ElastoPlasticMaterial


# Create a Mesher object to interface with GMSH. Provide a unique name. Multiple instance of this can be created.
myMeshModel = Mesher('myModel')

# Set the number of threads to use for any multi-threaded meshing algorithms e.g. HXT
myMeshModel.setNumThreads(4)
myMeshModel.setOptimiseNetgen(True)

# Set the meshing algorithm (optional) to use globally.
myMeshModel.setMeshingAlgorithm(pyccx.mesh.MeshingAlgorithm.FRONTAL_DELAUNAY)

# Add the geometry and assign a physical name 'PartA' which can reference the elements generated for the volume
myMeshModel.addGeometry('../models/cornerCube.step', 'PartA')

"""
Merges an assembly together. This is necessary there multiple bodies or volumes which share coincident faces. GMSH
will automatically stitch these surfaces together and create a shared boundary which is often useful performing
analyses where fields overlap (e.g. heat transfer). This should not be done if contact analysis is performed.
"""

myMeshModel.mergeGeometry()

# Optionally set hte name of boundary name using the GMSH geometry identities
myMeshModel.setEntityName((2,1), 'MySurface1')
myMeshModel.setEntityName((2,2), 'MySurface2')
myMeshModel.setEntityName((2,3), 'Bottom_Face')
myMeshModel.setEntityName((2,4), 'MySurface4')
myMeshModel.setEntityName((2,5), 'MySurface5')
myMeshModel.setEntityName((3,1), 'PartA')

# Set the size of the mesh
geomPoints = myMeshModel.getPointsFromVolume(1)
myMeshModel.setMeshSize(geomPoints, 0.5)  # MM

# Generate the mesh
myMeshModel.generateMesh()

# To preview the mesh call the command in the inbuilt GMSH Gui
# myMeshModel.showGui()

# Obtain the surface faces (normals facing outwards) for surface
surfFaces2 = myMeshModel.getSurfaceFacesFromSurfId(1) # MySurface1
bottomFaces = myMeshModel.getSurfaceFacesFromSurfId(3) # ('Bottom_Face

# Obtain all nodes associated with each surface
surface1Nodes = myMeshModel.getNodesFromEntity((2,1)) # MySurface
surface2Nodes = myMeshModel.getNodesFromEntity((2,2)) # MySurface2
surface4Nodes = myMeshModel.getNodesFromEntity((2,4)) # MySurface4
surface6Nodes = myMeshModel.getNodesFromEntity((2,6)) # MySurface6

# An alternative method is to get nodes via the surface
bottomFaceNodes = myMeshModel.getNodesFromSurfaceByName('Bottom_Face')

# or via general query
surface5Nodes = myMeshModel.getNodesByEntityName('MySurface5') # MySurface5

# Obtain nodes from the volume
volumeNodes = myMeshModel.getNodesFromVolumeByName('PartA')

import gmsh
#gmsh.fltk.run()

""" Create the analysis"""
# Set the number of simulation threads to be used by Calculix Solver across all analyses

Simulation.setNumThreads(4)
analysis = Simulation(myMeshModel)

# Optionally set the working the base working directory
analysis.setWorkingDirectory('.')

print('Calculix version: {:d}. {:d}'.format(*analysis.version()))

# Add the Node Sets For Attaching Boundary Conditions#
analysis.nodeSets.append(NodeSet('surface1Nodes', surface1Nodes))
analysis.nodeSets.append(NodeSet('surface2Nodes', surface2Nodes))
analysis.nodeSets.append(NodeSet('surface4Nodes', surface4Nodes))
analysis.nodeSets.append(NodeSet('surface5Nodes', surface5Nodes))
analysis.nodeSets.append(NodeSet('surface6Nodes', surface6Nodes))
analysis.nodeSets.append(NodeSet('bottomFaceNodes', bottomFaceNodes))
analysis.nodeSets.append(NodeSet('VolumeNodeSet', volumeNodes))


# ===============  Initial Conditions =============== #

analysis.initialConditions.append({'type': 'temperature', 'set': 'VolumeNodeSet', 'value': 0.0})

# ===============  Thermal Load Cases =============== #

# Create a thermal load case and set the timesettings
thermalLoadCase = LoadCase('Thermal Load Case')

# Set the loadcase type to thermal - eventually this will be individual classes
thermalLoadCase.setLoadCaseType(LoadCaseType.THERMAL)

# Set the thermal analysis to be a steadystate simulation
thermalLoadCase.isSteadyState = True
thermalLoadCase.setTimeStep(5.0, 5.0, 5.0)

# Attach the nodal and element result options to each loadcase
# Set the nodal and element variables to record in the results (.frd) file
nodeThermalPostResult = NodalResult('VolumeNodeSet')
nodeThermalPostResult.useNodalTemperatures = True

elThermalPostResult = ElementResult('Volume1')
elThermalPostResult.useHeatFlux = True

thermalLoadCase.resultSet = [nodeThermalPostResult, elThermalPostResult]

"""
class DOF(Enum):
    UX = 1
    V = 2
    W = 3
    T = 11
"""

# Set thermal boundary conditions for the loadcase(Current Version)

thermalLoadCase.boundaryConditions.append(
    {'type': 'fixed', 'nodes': 'surface6Nodes', 'dof': [11], 'value': [60]})

thermalLoadCase.boundaryConditions.append(
    {'type': 'fixed', 'nodes': 'surface1Nodes', 'dof': [11], 'value': [20]})

thermalLoadCase.boundaryConditions.append({'type': 'faceflux', 'faces':  bottomFaces, 'flux':50})

# =========== Material  =========== #
# Add a elastic material and assign it to the volume.
# Note ensure that the units correctly correspond with the geometry length scales

steelMat = ElastoPlasticMaterial('Steel')
steelMat.E = 210000.
steelMat.alpha_CTE = [25e-6, 23e-6, 24e-6]   # Thermal Expansion Coefficient
steelMat.density = 1.0    # Density
steelMat.cp =  1.0        # Specific Heat
steelMat.k = 1.0          # Thermal Conductivity

analysis.materials.append(steelMat)

# Assign the material the volume (use the part name set for geometry)
analysis.materialAssignments = [('PartA', 'Steel')]

# Set the loadcases used in sequential order
analysis.loadCases = [thermalLoadCase]

# Run the analysis
analysis.run()

# Open the results  file ('input') is currently the file that is generated by PyCCX
results = analysis.results()

# The call to read must be done to load all loadcases and timesteps from the results file
results.read()

# Obtain the nodal temperatures
nodalTemp = results.lastIncrement()['temp'][:, 1]

# Obtain the nodal coordinates and elements for further p
tetEls = myMeshModel.getElementsByType(ElementType.TET4)
