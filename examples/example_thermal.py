"""
This example demonstrates how to use PyCCX to perform a thermal analysis on a simple geometry.
"""

import pyccx

# Import related to meshing
from pyccx.mesh import Mesher, Ent
import pyccx.mesh.elements as Elements

# Imports related to the FEA Analysis
from pyccx.bc import Fixed, HeatFlux
from pyccx.analysis import Simulation, SolidMaterialAssignment
from pyccx.core import DOF, ElementSet, NodeSet, SurfaceSet
from pyccx.results import ElementResult, NodalResult
from pyccx.loadcase import  LoadCase, LoadCaseType
from pyccx.material import ElastoPlasticMaterial

""" ====================== Meshing ====================== """
# Create a Mesher object to interface with GMSH. Provide a unique name. Multiple instance of this can be created.
myMeshModel = Mesher('myModel')

# Set the number of threads to use for any multi-threaded meshing algorithms e.g. HXT
myMeshModel.setNumThreads(4)
myMeshModel.setOptimiseNetgen(True)

# Set the meshing algorithm (optional) to use globally.
myMeshModel.set2DMeshingAlgorithm(pyccx.mesh.MeshingAlgorithm2D.DELAUNAY)
myMeshModel.set3DMeshingAlgorithm(pyccx.mesh.MeshingAlgorithm3D.FRONTAL_DELAUNAY)

# Add the geometry and assign a physical name 'PartA' which can reference the elements generated for the volume
myMeshModel.addGeometry('../models/cornerCube.step', 'PartA')

"""
Merges an assembly together. This is necessary there multiple bodies or volumes which share coincident faces. GMSH
will automatically stitch these surfaces together and create a shared boundary which is often useful performing
analyses where fields overlap (e.g. heat transfer). This should not be done if contact analysis is performed.
"""

myMeshModel.mergeGeometry()

# Optionally set hte name of boundary name using the GMSH geometry identities
myMeshModel.setEntityName((Ent.Surface,1), 'MySurface1')
myMeshModel.setEntityName((Ent.Surface,2), 'MySurface2')
myMeshModel.setEntityName((Ent.Surface,3), 'Bottom_Face')
myMeshModel.setEntityName((Ent.Surface,4), 'MySurface4')
myMeshModel.setEntityName((Ent.Surface,5), 'MySurface5')
myMeshModel.setEntityName((Ent.Volume,1), 'PartA')

"""
Set the average element size of the mesh. This requires using a mesh control which
is applied to the geometry points available within the Volume. GMSH does not natively
support applying seeds along edges like commercial software.
"""
geomPoints = myMeshModel.getPointsFromVolume(1)
myMeshModel.setMeshSize(geomPoints, 0.5)  # [MM]

# Generate the mesh
myMeshModel.generateMesh()

# Obtain the surface faces (normals facing outwards) for surface
surfFaces2 = myMeshModel.getSurfaceFacesFromSurfId(1)  # MySurface1
bottomFaces = myMeshModel.getSurfaceFacesFromSurfId(3) # Bottom_Face

# Obtain all nodes associated with each surface
surface1Nodes = myMeshModel.getNodesFromEntity((Ent.Surface,1)) # MySurface
surface2Nodes = myMeshModel.getNodesFromEntity((Ent.Surface,2)) # MySurface2
surface4Nodes = myMeshModel.getNodesFromEntity((Ent.Surface,4)) # MySurface4
surface6Nodes = myMeshModel.getNodesFromEntity((Ent.Surface,6)) # MySurface6

# An alternative method is to get nodes via the surface
bottomFaceNodes = myMeshModel.getNodesFromSurfaceByName('Bottom_Face')

# or via general query
surface5Nodes = myMeshModel.getNodesByEntityName('MySurface5') # MySurface5

# Obtain nodes from the volume
volumeNodes = myMeshModel.getNodesFromVolumeByName('PartA')

# The generated mesh can be interactively viewed natively within gmsh by calling the following
#myMeshModel.showGui()

""" Create the analysis"""
# Set the number of simulation threads to be used by Calculix Solver across all analyses

Simulation.setNumThreads(4)

# Set the direct path to the Calculix executable
Simulation.CALCULIX_PATH = '/opt/homebrew/Cellar/calculix-ccx/2.22/bin/ccx_2.22'

# Create a Simulation object based on the supplied mesh model
analysis = Simulation(myMeshModel)

# Optionally set the working the base working directory
analysis.setWorkingDirectory('.')

print('Calculix version: {:d}. {:d}'.format(*analysis.version()))

# Add the Node Sets For Attaching Boundary Conditions #
# Note a unique name must be provided
surface1NodeSet = NodeSet('surface1Nodes', surface1Nodes)
surface2NodeSet = NodeSet('surface2Nodes', surface2Nodes)
surface4NodeSet = NodeSet('surface4Nodes', surface4Nodes)
surface5NodeSet = NodeSet('surface5Nodes', surface5Nodes)
surface6NodeSet = NodeSet('surface6Nodes', surface6Nodes)
bottomFaceNodeSet = NodeSet('bottomFaceNodes', bottomFaceNodes)
volNodeSet = NodeSet('VolumeNodeSet', volumeNodes)

# Create an element set using a concise approach

partElSet = ElementSet('PartAElSet', myMeshModel.getElementIds((Ent.Volume,1)))

# Create a surface set
bottomFaceSet = SurfaceSet('bottomSurface', bottomFaces)

# Add the userdefined nodesets to the analysis
analysis.nodeSets = [surface1NodeSet, surface2NodeSet, surface4NodeSet, surface5NodeSet, surface6NodeSet,
                     bottomFaceNodeSet, volNodeSet]


"""
Set the element types
The element types can be set by the user to ensure that the correct element type is used for the analysis.
"""

modelElIds = myMeshModel.getElementIds((Ent.Volume, Ent.All))
myMeshModel.setMeshAssignmentsByType(modelElIds, Elements.TET4)

# Check if there are any  elements that have not been assigned an element type.
if len(myMeshModel.identifyUnassignedElements()) > 0:
    raise Exception("identified unassigned elements")


"""
Initial Conditions 
-----------------
Initial conditions can be set for the analysis. This is used for setting the nodal temeprature
at the zeroth increment of the analysis.

When setting the initial condition,when using a NodalSet, either a NodeSet name, or object can be 
specified or an explicit list of nodes to apply the initial value upon.
"""

analysis.initialConditions.append({'type': 'temperature', 'set': 'VolumeNodeSet', 'value': 0.0})

""" 
Thermal Load Cases
-----------------
In this example a thermal load case is defined to simulate a transient heat transfer problem in a
simple geometry for a laser optic model.
"""

# Create a thermal load case and set the timesettings
thermalLoadCase = LoadCase('Thermal_Load_Case')

# Set the loadcase type to thermal - eventually this will be individual classes
thermalLoadCase.setLoadCaseType(LoadCaseType.THERMAL)

# Set the thermal analysis to be a steadystate simulation
thermalLoadCase.isSteadyState = False
thermalLoadCase.setTimeStep(0.5, 0.5, 5.0)

"""
Results Export
-----------------
Attach the nodal and element result options to each specific loadcase
The results will be exported to the .frd file
"""

# Set the nodal and element variables to record in the results
# When specifying the result node set, if the node set is not specified, all nodes will be used
nodeThermalPostResult = NodalResult()
nodeThermalPostResult.temperature = True

elThermalPostResult = ElementResult(partElSet)
elThermalPostResult.heatFlux = True

thermalLoadCase.resultSet = [nodeThermalPostResult, elThermalPostResult]

# Set thermal boundary conditions for the loadcase
thermalLoadCase.boundaryConditions = [Fixed(surface6NodeSet,  dof=[DOF.T], values = [60.0]),
                                      Fixed(surface1NodeSet,  dof=[DOF.T], values = [20.0]),
                                      HeatFlux(bottomFaceSet, flux=50.0)]

"""
====================== Materials  ======================
Material should be defined before the material assignment. This can include additional
thermal and mechanical properties that are used for each subsequent analysis load step.

- Add a elasto-plastic material and assign it to the volume.
- Note: Ensure that the units correctly correspond with the geometry length scales
"""

steelMat = ElastoPlasticMaterial('Steel')
steelMat.E = 210000.      # [MPa] Young's Modulus
steelMat.alpha_CTE = [25e-6, 23e-6, 24e-6]  # Thermal Expansion Coefficient
steelMat.density = 1.0    # Density
steelMat.cp =  1.0        # Specific Heat
steelMat.k = 1.0          # Thermal Conductivity

analysis.materials.append(steelMat)

"""
Material Assignment
--------------------
The material is assigned to the element set created earlier for the part

- SolidMaterialAssignment is used to assign a material to 3D continuum elements used in the model
"""

analysis.materialAssignments = [
    SolidMaterialAssignment("solid_material", elementSet=partElSet, material=steelMat)
]

# Set the loadcases used in sequential order
analysis.loadCases = [thermalLoadCase]

""" ====================== Analysis Run  ====================== """
# Run the analysis
try:
    analysis.run()
except:
    print('Analysis failed on this design')

""" ====================== Post-Processing ====================== """
# Open the results  file ('input') is currently the file that is generated by PyCCX
results = analysis.results()

# The call to read must be done to load all loadcases and timesteps from the results file
results.read()

# Obtain the nodal temperatures
nodalTemp = results.lastIncrement()['temp'][:, 1]

# Obtain the nodal coordinates and elements for further p
tetEls = myMeshModel.getElementsByType(Elements.TET4)

import pyccx.utils.exporters as exporters

# Export the results to VTK format as a significant timestep for post-processing
exporters.exportToVTK('result.vtu', results, inc=-1)

# Export the results to PVD format for visualization in Paraview - this includes all the timesteps during the analysis
exporters.exportToPVD('results.pvd', results)
