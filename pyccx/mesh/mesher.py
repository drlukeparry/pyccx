from abc import ABC, abstractmethod
from enum import Enum, IntEnum
from typing import Any, List, Optional, Tuple

import logging
import os
import numpy as np
import gmsh

from . import elements
from .utils import *


class MeshingAlgorithm2D(IntEnum):
    MESHADAPT    = 1
    AUTO         = 2
    INITIAL_ONLY = 3
    DELAUNAY     = 5
    FRONTAL      = 6
    BAMG         = 7
    FRONTAL_QUAD = 8
    PACK_PRLGRMS = 9
    PACK_PRLGRMS_CSTR = 10
    QUAD_QUASI_STRUCT = 11

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


class MeshingAlgorithm3D(IntEnum):
    DELAUNAY = 1
    FRONTAL = 4
    FRONTAL_DELAUNAY = 5
    FRONTAL_HEX = 6
    MMG3D = 7
    RTREE = 9
    HXT = 10

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


class RecombinationAlgorithm(IntEnum):
    SIMPLE = 0
    BLOSSOM = 1
    SIMPLE_QUAD = 2
    BLOSSOM_QUAD = 3


class Mesher:
    """
    The Mesher class provides the base interface built upon the GMSH-SDK API operations. It provides the capability
    to mesh multiple PythonOCC objects
    """

    """ Minimum tolerance used for truncating nodal coordiante values """
    EPSILON = 1E-10

    # Static class variables for meshing operations
    ElementOrder = 1
    NumThreads = 4
    OptimiseNetgen = True
    Units = ''
    Initialised = False

    # Instance methods
    def __init__(self, modelName: str):
        """
        :param modelName: str: A model name is required for GMSH
        """

        # The GMSH Options must be initialised before performing any operations
        Mesher.initialise()

        # Create an individual model per class
        self._modelName = modelName
        self.modelMeshSizeFactor = 0.05  # 5% of model size
        self.geoms = []
        self.surfaceSets = []
        self.edgeSets = []
        self._meshAssignments = []

        self._isMeshGenerated = False
        self._isDirty = False # Flag to indicate model hasn't been generated and is dirty
        self._isGeometryDirty = False

        self._recombinationAlgorithm = RecombinationAlgorithm.BLOSSOM
        self._meshingAlgorithm2D = MeshingAlgorithm2D.DELAUNAY
        self._meshingAlgorithm3D = MeshingAlgorithm3D.DELAUNAY

        # Set the model name for this instance
        gmsh.model.add(self._modelName)

    @classmethod
    def showGui(cls):
        """
        Opens up the native GMSH Gui to inspect the geometry in the model and the mesh. This will block the Python script
        until the GUI is exited.
        """
        if cls.Initialised:
            gmsh.fltk.run()

    @property
    def version(self) -> Tuple[int, int, int]:
        """ Version of the GMSH SDK available """

        return gmsh.GMSH_API_VERSION_MAJOR, gmsh.GMSH_API_VERSION_MINOR, gmsh.GMSH_API_VERSION_PATCH

    def open(self, filename: str):
        """
        Opens a GMSH file and loads the model into the current instance

        :param filename: The filename of the GMSH file
        """

        self.setAsCurrentModel()

        try:
            gmsh.open(filename)

        except:
            raise Exception('Unable to open GMSH file ({:s})'.format(filename))

        self._modelName = gmsh.model.getCurrent()

        self.clearMeshAssignments()
        self._isMeshGenerated = True

    def clearPhysicalGroups(self, dimension: int = None):
        """
        Clears all physical groups in the model
        :param dimension: Integer is the dimension of the physical group to clear. Default is None.
        """

        self.setAsCurrentModel()

        if not dimension:
            gmsh.model.removePhysicalGroups()
        else:
            gmsh.model.removePhysicalGroups(dimension)

    def clearMeshAssignments(self, elType = None) -> None:
        """
        Clears the mesh assignments for the model

        :param elType: The element type to clear the assignments
        """
        if elType is None:
            self._meshAssignments = {}
        else:
            self._meshAssignments[elType] = []


    def setMeshAssignmentsById(self, elIds, elType: elements.BaseElementType) -> None:
        """
        Set the element type and concatenate elements to this

        :param elType:
        :param el:
        :return:
        """
        if not self._isMeshGenerated:
            logging.error('Mesh is currently not generated')
            return

        # Iterate across existing mesh assignments and remove duplicate elements
        for key, value in self._meshAssignments.items():
            self._meshAssignments[key] = np.delete(value, np.argwhere(np.isin(value, elIds).ravel()))

        elSet = {}
        for eId, eType in zip(elIds, elType):
            eList = elSet.get(eType,[])
            eList.append(eId)

        for eType, eIds in elSet.items():
            # Obtain a list of elements
            elSet = self._meshAssignments.get(eType, [])
            elSet = np.unique(np.hstack([elSet, eIds])).astype(np.int64)

            self._meshAssignments[elType] = elSet

        return


    def setMeshAssignmentsByType(self, elIds: np.array, elType: elements.BaseElementType) -> None:
        """
        Set the element type and concatenate elements to this

        :param elType:
        :param el:
        :return:
        """
        if not self._isMeshGenerated:
            logging.error('Mesh is currently not generated')
            return

        el = np.asanyarray(elIds)
        # Iterate across existing mesh assignments and remove duplicate elements
        for key, value in self._meshAssignments.items():
            self._meshAssignments[key] = np.delete(value, np.argwhere(np.isin(value, el).ravel()))

        # Obtain a list of elements
        if elType:
            elSet = self._meshAssignments.get(elType, [])
            elSet = np.unique(np.hstack([elSet, el])).astype(np.int64)

            self._meshAssignments[elType] = elSet

        return


    def getAllPhysicalGroupElements(self):

        self.setAsCurrentModel()

        if not self._isMeshGenerated:
            raise Exception('Mesh is not generated')

        tags = gmsh.model.getPhysicalGroups()
        entTags = []

        for x in tags:
            eTags = gmsh.model.getEntitiesForPhysicalGroup(x[0], x[1])
            for eTag in eTags:
                entTags.append((x[0], eTag))

        elIdList = []

        for eTag in entTags:
            elIdList.append(self.getElements(eTag))

        return self._mergeElements(elIdList)

    @staticmethod
    def _mergeElements(elList):
        """
        Internal method for merging list of elements of different types

        :return: A dictionary consisting of element types as keys and elements that are compounded together
        """
        fndElements = elList

        eOut = {}

        for els in fndElements:

            for i, eType in enumerate(els[0]):
                if not eOut.get(eType, False):
                    eOut[eType] = [[], []]

                eOut[eType][0].append(els[1][i])
                eOut[eType][1].append(els[2][i])

        for id in eOut.keys():
            elType = elements.getElementById(id)

            eOut[id][0] = np.hstack(eOut[id][0]).ravel()
            eOut[id][1] = np.vstack(eOut[id][1]).reshape(-1, elType.nodes)

        # process the keys so that the dimensions are consistent with the formatting of the element types
        return eOut

    @staticmethod
    def _restructureElementStructure(els):
        """
        Restructures the internal gmsh element structures so the element dimensions are consistent with the element types

        :param els: The GMSH element type
        :return: The restructured element list
        """

        for i, id in enumerate(els[0]):
            elType = elements.getElementById(id)
            els[2][i] = els[2][i].reshape(-1, elType.nodes)

            # process the keys so that the dimensions are consistent with the formatting of the element types
        return els

    def identifyUnassignedElements(self) -> np.array:
        """
        Returns the list of elements that have not been assigned element types

        :return: List of elements that have no element type designated
        """
        self.setAsCurrentModel()

        # Obtain the current list of elements across the entire model
        elIds = np.hstack(self.getAllPhysicalGroupElements())
        fndElIds = np.sort(elIds)

        meshAssignmentVals = self._meshAssignments.values()

        if len(meshAssignmentVals) == 0:
            return fndElIds

        # Concatenate all the list of assigned elements
        assignedElIds = np.hstack([x for x in self._meshAssignments.values()])

        # Identify the elements that have not been assigned
        out = fndElIds[np.isin(fndElIds, assignedElIds, invert=True)]

        return out

    @property
    def physicalGroups(self):
        """
        Returns the physical groups tags available in the model
        """
        self.setAsCurrentModel()

        tags = gmsh.model.getPhysicalGroups()
        return tags

    def maxPhysicalGroupId(self, dim: int) -> int:
        """
        Returns the highest physical group id in the GMSH model

        :param dim: int: The chosen dimension
        :return: int: The highest group id used for the chosen dimension
        """
        self.setAsCurrentModel()

        physicalGroups = gmsh.model.getPhysicalGroups(dim)

        if len(physicalGroups) > 0:
            return np.max([x[1] for x in gmsh.model.getPhysicalGroups(dim)])
        else:
            return 0

    def getVolumeName(self, volId: int) -> str:
        """
        Gets the volume name (if assigned)

        :param volId: int: Volume id of a region
        """
        self.setAsCurrentModel()

        return gmsh.model.getEntityName(Ent.Volume, volId)

    def getEntityName(self, id: Tuple[int, int]) -> str:
        """
        Returns the name of an entity given an id (if assigned)
        :param id: Dimension, Entity Id

        """
        self.setAsCurrentModel()

        return gmsh.model.getEntityName(id[0],id[1])

    def setEntityName(self, id: Tuple[int, int], name: str) -> None:
        """
        Set the geometrical entity name - useful only as reference when viewing in the GMSH GUI

        :param id: Entity Dimension and Entity Id
        :param name: The entity name

        """
        self.setAsCurrentModel()
        gmsh.model.setEntityName(id[0], id[1], name)

    def setEdgePhysicalName(self, edgeId: Any, name: str) -> int:
        """
        Sets the Physcal Group Name of the Curve(s)

        :param surfId: The set of curve ids
        :param name: Name assigned to volume
        """

        if not isinstance(edgeId, list):
            surfId = [edgeId]

        self.setAsCurrentModel()

        maxId = self.maxPhysicalGroupId(Ent.Curve)
        gId = gmsh.model.addPhysicalGroup(Ent.Curve, edgeId, maxId+1, name)

        return gId

    def setSurfacePhysicalName(self, surfId: Any, name: str) -> int:
        """
        Sets the Physical Group Name of the Surfaces(s)

        :param surfId: The set of surface ids
        :param name: Name assigned to volume
        """

        if not isinstance(surfId, list):
            surfId = [surfId]

        self.setAsCurrentModel()
        maxId = self.maxPhysicalGroupId(Ent.Surface)
        gId = gmsh.model.addPhysicalGroup(Ent.Surface, surfId, maxId+1, name)

        return gId

    def setVolumePhysicalName(self, volId: Any, name: str) -> int:
        """
        Sets the Physical Group Name of the Volumes(s)

        :param volId: The set of volume ids
        :param name: The name assigned to volume group
        """

        if not isinstance(volId, list):
            surfId = [volId]

        self.setAsCurrentModel()
        maxId = self.maxPhysicalGroupId(3)

        gId = gmsh.model.addPhysicalGroup(Ent.Volume, volId, maxId+1, name)
        return gId

    def setEntityPhysicalName(self, id: Tuple[int,int], name: str) -> int:
        """
        Sets the geometric name of the volume id

        :param id: Dimension, Entity Id tuple(int, int):
        :param name: str: Name assigned to entity
        """
        self.setAsCurrentModel()
        maxId = self.maxPhysicalGroupId(id[0])
        gId = gmsh.model.addPhysicalGroup(id[0], [id[1]], maxId+1, name)

        return gId

    @property
    def name(self) -> str:
        """
        The name of the GMSH model
        """
        return self._modelName

    def isDirty(self) -> bool:
        """
        Has the model been successfully generated and no pending modifications exist.
        """
        return self._isDirty

    def setModelChanged(self, state: bool = False) -> None:
        """
        Any changes to GMSH model should call this to prevent inconsistency in a generated model

        :param state:  Force the model to be shown as generated
        """

        self._isDirty = state

    def addGeometry(self, filename: str, name: str, meshFactor: Optional[float] = 0.03):
        """
        Adds CAD geometry into the GMSH kernel. The filename of compatiable model files along with the mesh factor
        should be used to specify a target mesh size.

        :param filename:
        :param name: Name to assign to the geometries imported
        :param meshFactor: Initialise the target element size to a proportion of the average bounding box dimensions
        """

        if not (os.path.exists(filename) and os.access(filename, os.R_OK)):
            raise ValueError('File ({:s}) is not readable'.format(filename))

        # Adds a single volume
        self.setAsCurrentModel()

        # Additional geometry will be merged into the current  model
        gmsh.merge(filename)
        self.geoms.append({'name': name, 'filename': filename, 'meshSize': None, 'meshFactor': meshFactor})

        # Set the name of the volume
        # This automatically done to ensure that are all exported. This may change to parse through all volumes following
        # merged and be recreated

        gmsh.model.setEntityName(Ent.Volume, len(self.geoms), name)
        gmsh.model.addPhysicalGroup(Ent.Volume, [len(self.geoms)], len(self.geoms))
        gmsh.model.setPhysicalName(Ent.Volume, len(self.geoms), name)

        # set the mesh size for this geometry
        bbox = self.getGeomBoundingBoxById(len(self.geoms))
        extents = bbox[1, :] - bbox[0, :]
        avgDim = np.mean(extents)
        meshSize = avgDim * meshFactor

        logging.info('GMSH: Avg dim', avgDim, ' mesh size: ', meshSize)
        geomPoints = self.getPointsFromVolume(len(self.geoms))

        # Set the geometry volume size
        self.geoms[-1]['meshSize'] = meshSize

        self.setMeshSize(geomPoints, meshSize)

        self._isGeometryDirty = True
        self.setModelChanged()

    def clearMesh(self) -> None:
        """
        Clears any meshes previously generated by GMSH
        """

        self.setAsCurrentModel()

        logging.info('Clearing mesh \n')

        gmsh.model.mesh.clear()

    @property
    def volumes(self) -> List[int]:
        """
        The ids for all volume geometry in the model
        """
        self.setAsCurrentModel()
        tags = gmsh.model.getEntities(Ent.Volume)
        return [(x[1]) for x in tags]

    @property
    def surfaces(self) -> List[int]:
        """
        The ids for all surface geometry in the model
        """
        self.setAsCurrentModel()
        tags = gmsh.model.getEntities(Ent.Surface)
        return [(x[1]) for x in tags]

    @property
    def edges(self) -> List[int]:
        """
        The ids for all edge geometry in the model
        """
        tags = gmsh.model.getEntities(Ent.Curve)
        return [(x[1]) for x in tags]

    @property
    def points(self) -> List[int]:
        """
        The ids for all point geometry in the model
        """
        self.setAsCurrentModel()
        tags = gmsh.model.getEntities(Ent.Point)
        return [(x[1]) for x in tags]

    @property
    def meshAssignments(self):
        return self._meshAssignments

    def mergeGeometry(self) -> None:
        """
        Geometry is merged/fused together. Coincident surfaces and points are automatically merged together, which
        enables a coherent mesh to be generated when these align exactly.
        """

        self.setAsCurrentModel()

        volTags = gmsh.model.getEntities(Ent.Volume)
        out = gmsh.model.occ.fragment(volTags[1:], [volTags[0]])

        # Synchronise the model
        gmsh.model.occ.synchronize()

        self._isGeometryDirty = True
        self.setModelChanged()

    def boundingBox(self) -> np.array:
        """
        Returns the bounding box of the model

        :return: The bounding box of the GMSH Model
        """
        self.setAsCurrentModel()
        return np.array(gmsh.model.getBoundingBox())

    def getGeomBoundingBoxById(self, tagId: int) -> np.array:
        """
        Returns the bounding box of the geometry (volume) by id

        :param tagId: The geometry id
        :return: The bounding box of the GMSH Model
        """
        self.setAsCurrentModel()
        return np.array(gmsh.model.getBoundingBox(Ent.Volume, tagId)).reshape(-1, 3)

    def getGeomBoundingBoxByName(self, volumeName):
        self.setAsCurrentModel()
        volTagId = self.getIdByVolumeName(volumeName)
        return np.array(gmsh.model.getBoundingBox(Ent.Volume, volTagId)).reshape(-1, 3)

    def setMeshSize(self, pnts: List[int], size: float) -> None:
        """
        Sets the mesh element size along an entity, however, this can only currently be performed using

        :param pnts: Point ids to set the mesh size
        :param size: Set the desired mesh length size at this point
        """

        self.setAsCurrentModel()
        tags = [(0, x) for x in pnts]
        gmsh.model.mesh.setSize(tags, size)

        self.setModelChanged()

    def set3DMeshingAlgorithm(self, meshingAlgorithm: MeshingAlgorithm3D) -> None:
        """
        Sets the meshing algorithm to use by GMSH for the model
        
        :param meshingAlgorithm: The selected 3D Meshing Algorithm used
        """

        # The meshing algorithm is applied before generation, as this may be model specific
        self._meshingAlgorithm3D = meshingAlgorithm
        self.setModelChanged()

    def set2DMeshingAlgorithm(self, meshingAlgorithm: MeshingAlgorithm2D) -> None:
        """
        Sets the meshing algorithm to use by GMSH for the model

        :param meshingAlgorithm: The selected 2D Meshing Algorithm used
        """

        # The meshing algorithm is applied before generation, as this may be model specific
        self._meshingAlgorithm2D = meshingAlgorithm
        self.setModelChanged()


    def setRecombinationAlgorithm(self, recombinationAlgorithm: RecombinationAlgorithm) -> None:
        """
        Sets the recombination algorithm to use by GMSH for the model

        :param recombinationAlgorithm: The selected recombination algorithm for use
        """

        # The meshing algorithm is applied before generation, as this may be model specific
        self._recombinationAlgorithm = recombinationAlgorithm

        self.setModelChanged()

    ## Class Methods
    @classmethod
    def setUnits(cls, unitVal):
        cls.Units = unitVal

        if cls.Initialised:
            gmsh.option.setString("Geometry.OCCTargetUnit", Mesher.Units)

    @classmethod
    def setElementOrder(cls, elOrder: int):
        """
        Sets the element order globally for the entire model. Note that different element orders cannot be used within
        the same GMSH model during generation

        :param elOrder: The element order
        """
        cls.ElementOrder = elOrder

        if cls.Initialised:
            gmsh.option.setNumber("Mesh.ElementOrder", Mesher.ElementOrder)

    @classmethod
    def setOptimiseNetgen(cls, state: bool) -> None:
        """
        Sets an option for GMSH to internally use Netgen to optimise the element quality of the generated mesh. Enabled
        by default.

        :param state: Toggles the option
        """
        cls.OptimiseNetgen = state

        if cls.Initialised:
            gmsh.option.setNumber("Mesh.OptimizeNetgen", (1 if cls.OptimiseNetgen else 0))

    @classmethod
    def getNumThreads(cls) -> int:
        """
        Gets the number of threads used for parallel processing by GMSH
        """
        return cls.NumThreads

    @classmethod
    def setNumThreads(cls, numThreads: int) -> None:
        """
        Sets the number of threads to be used for parallel processing by GMSH

        :param numThreads:
        """
        cls.NumThreads = numThreads

        if cls.Initialised:
            gmsh.option.setNumber("Mesh.MaxNumThreads2D", Mesher.NumThreads)
            gmsh.option.setNumber("Mesh.MaxNumThreads3D", Mesher.NumThreads)

    @classmethod
    def setMeshSizeFactor(self, meshSizeFactor: float) -> None:
        """
        The mesh factor size provides an estimate length for the initial element sizes based on proportion of the maximumum
        bounding box length.

        :param meshSizeFactor: The mesh factor size between [0.,1.0]
        """

        if meshSizeFactor > 1.0 or meshSizeFactor < 1e-8:
            raise ValueError('Invalid size for the mesh size factor was provided')

        self.modelMeshSizeFactor = meshSizeFactor

    @classmethod
    def finalize(cls):
        gmsh.finalize()
        cls.Initialised = False

    @classmethod
    def initialise(cls) -> None:
        """
        Initialises the GMSH runtime and sets default options. This is called automatically once.
        """

        if cls.Initialised:
            return

        logging.info('Initialising GMSH \n')
        gmsh.initialize()

        # Mesh.Algorithm3
        # Default value: 1#
        gmsh.option.setNumber("Mesh.Algorithm", MeshingAlgorithm3D.FRONTAL_DELAUNAY)
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", RecombinationAlgorithm.BLOSSOM)
        #        gmsh.option.setNumber("Mesh.Algorithm3D", 10);

        gmsh.option.setNumber("Mesh.ElementOrder", Mesher.ElementOrder)
        #        gmsh.option.setNumber("Mesh.OptimizeNetgen",1)
        gmsh.option.setNumber("Mesh.MaxNumThreads2D", Mesher.NumThreads)
        gmsh.option.setNumber("Mesh.MaxNumThreads3D", Mesher.NumThreads)

        # Set to use incomplete 2nd order elements
        gmsh.option.setNumber('Mesh.SecondOrderIncomplete', 1)

        # gmsh.option.setNumber("Mesh.SaveGroupsOfNodes", 1);
        # gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 15);

        gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
        gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 1)
        gmsh.option.setNumber("Mesh.SaveAll", 0)

        # OCC Options
        gmsh.option.setNumber("Geometry.OCCFixDegenerated", 1)
        gmsh.option.setString("Geometry.OCCTargetUnit", Mesher.Units)

        # General Options
        gmsh.option.setString("General.ErrorFileName", 'error.log')
        gmsh.option.setNumber("General.Terminal", 1)

        ########## Gui Options ############
        gmsh.option.setNumber("General.Antialiasing", 1)
        gmsh.option.setNumber("Geometry.SurfaceType", 2)

        # Discretisation for high-order elements when visualised
        gmsh.option.setNumber("Mesh.NumSubEdges", 4)

        # The following GMSH options do not appear to change anything
        # Import labels is required inorder to correctly reference geometric
        # entities and their associated mesh entities
        gmsh.option.setNumber("Geometry.OCCImportLabels", 1)

        # gmsh.option.setNumber("Geometry.OCCAutoFix", 0)
        cls.Initialised = True

    def setAsCurrentModel(self):
        """
        Sets the current model to that specified in the class instance because
        Only one instance of GMSH sdk is available so this must be
        dynamically switched between models for multiple instances of a Mesher.
        """

        gmsh.model.setCurrent(self._modelName)

    def removeEdgeMeshes(self) -> None:
        """
        Removes edges (1D mesh entities) from the GMSH model
        """

        self.setAsCurrentModel()

        tags = gmsh.model.getPhysicalGroups(Ent.Curve)

        for tag in tags:
            # Remove all tri group surfaces
            logging.info('removing edge {:s}'.format(gmsh.model.getPhysicalName(1, tag[1])))
            gmsh.model.removePhysicalGroups(tag)

        self.setModelChanged()

    def removeSurfaceMeshes(self):
        """
        Removes surface meshes (2D mesh entities) from the GMSH model
        """

        self.setAsCurrentModel()

        tags = gmsh.model.getPhysicalGroups(Ent.Surface)

        for tag in tags:
            # Remove all tri group surfaces
            print('removing surface {:s}'.format(gmsh.model.getPhysicalName(2, tag[1])))
            gmsh.model.removePhysicalGroups([tag])

        self.setModelChanged()

    # Geometric methods
    def getPointsFromVolume(self, id: int) -> List[int]:
        """
        From a Volume Id, obtain all Point Ids associated with this volume - note may include shared points.

        :param id: Volume ID
        :return: list(int) - List of Point Ids
        """
        self.setAsCurrentModel()

        pnts = gmsh.model.getBoundary([(Ent.Volume, id)], recursive=True)
        return [x[1] for x in pnts]

    def getPointsFromEntity(self, id: Tuple[int,int]) -> List[int]:
        """
        From an Id, obtain all Point Ids associated with this volume - note may include shared points.

        :param id: Dimension and Entity ID
        :return: List of Point Ids
        """
        self.setAsCurrentModel()
        pnts = gmsh.model.getBoundary([id], recursive=True)
        return [x[1] for x in pnts]

    def getChildrenFromEntities(self, id: Tuple[int,int]) -> List[int]:
        """
        From a Entity, obtain all children associated with this volume - note may include shared entities.

        :param id:  Dimension, Entity Id
        :return: List of Ids
        """
        self.setAsCurrentModel()

        entities = gmsh.model.getBoundary([id], recursive=False)
        return [x[1] for x in entities]

    def getSurfacesFromVolume(self, id: int) -> List[int]:
        """
        From a Volume Id, obtain all Surface Ids associated with this volume - note may include shared boundary surfaces.

        :param id:  Volume Id
        :return: List of surface Ids
        """

        self.setAsCurrentModel()
        surfs = gmsh.model.getBoundary( [(Ent.Volume, id)], recursive=False)
        return [x[1] for x in surfs]

    def getPointsFromVolumeByName(self, volumeName: str):
        """
        Returns all geometric points from a given volume domain by its name (if assigned)
        :param volumeName: volumeName
        :return:
        """
        return self.getPointsFromVolume(self.getIdBySurfaceName(volumeName))

    # Mesh methods
    def getNodeIds(self) -> np.array:
        """
        Returns the nodal ids from the entire GMSH model
        :return:
        """
        self.setAsCurrentModel()

        if not self._isMeshGenerated:
            raise Exception('Mesh is not generated')

        nodeIds = gmsh.model.mesh.getNodes()[0]
        nodeIds = np.sort(nodeIds)
        return nodeIds

    def getNodes(self):
        """
        Returns the nodal coordinates from the entire GMSH model. These are sorted automatically by the node-id

        :return: The nodal coordinates to the correspodning node ids
        """
        self.setAsCurrentModel()

        if not self._isMeshGenerated:
            raise Exception('Mesh is not generated')

        nodeVals= gmsh.model.mesh.getNodes()

        nodeCoords = nodeVals[1].reshape(-1, 3)

        nodeCoordsSrt = nodeCoords[np.argsort(nodeVals[0]-1)]
        return nodeCoordsSrt


    def getElementIds(self, entityId: Tuple[int,int] = None, merge: bool = True):
        """
        Returns the elements for the entire model or optionally a specific entity.
        :param entityId: The entity id to obtain elements for
        :param merge: Merge the element ids into a single array
        :return: A Tuple of  element types, element ids and corresponding node ids
        """
        self.setAsCurrentModel()

        if not self._isMeshGenerated:
            raise Exception('Mesh is not generated')

        result = None
        if entityId:
            # return all the elements for the entity
            result = gmsh.model.mesh.getElements(entityId[0], entityId[1])

        else:
            # Return all the elements in the model
            result = gmsh.model.mesh.getElements()

        if merge:
            return np.hstack(result[1]).ravel()
        else:
            return result[1]


    def getElements(self, entityId: Tuple[int,int] = None):
        """
        Returns the elements for the entire model or optionally a specific entity.

        :param entityId: The entity id to obtain elements for
        :return: A Tuple of  element types, element ids and corresponding node ids
        """
        self.setAsCurrentModel()

        if not self._isMeshGenerated:
            raise Exception('Mesh is not generated')

        if entityId:
            # return all the elements for the entity
            result = gmsh.model.mesh.getElements(entityId[0], entityId[1])
        else:
            # Return all the elements in the model
            result = gmsh.model.mesh.getElements()

        result = self._restructureElementStructure(result)
        return result

    def getElementsByType(self, elType: elements.BaseElementType, elTag: Tuple[int,int] = None,
                        returnElIds: Optional[bool] = False) -> np.ndarray:
        """
        Returns all elements of type (elType) from the GMSH model, within class ElementTypes. Note: the element ids are returned with
        an index starting from 1 - internally GMSH uses an index starting from 1, like most FEA pre-processors

        :param elType: Element type
        :return: List of Element Ids
        """

        self.setAsCurrentModel()

        if not self._isMeshGenerated:
            raise Exception('Mesh is not generated')

        if elTag:
            elVar = gmsh.model.mesh.getElementsByType(elType.id, elTag)
        else:
            elVar = gmsh.model.mesh.getElementsByType(elType.id)

        elements = elVar[1].reshape(-1, elType.nodes)

        if returnElIds:
            return elVar[0], elements
        else:
            return elements

    def getElementsByPhysicalGroups(self, physicalGroup) -> np.ndarray:
        """
        Returns all elements associated with a physical group

        :param physicalGroup: The physical group id
        :return: The element ids associated with the physical groups
        """
        self.setAsCurrentModel()

        if not self._isMeshGenerated:
            raise Exception('Mesh is not generated')

        if not isinstance(physicalGroup, list):
            physicalGroup = [physicalGroup]

        # Obtain the surface-element physical groups and their associative element ids

        # Obtain the volume-element physical groups and their associative element ids
        eTags = [(x[0], gmsh.model.getEntitiesForPhysicalGroup(x[0], x[1])) for x in physicalGroup]

        entTags = []
        for tag in eTags:
            entTags += [(tag[0], x) for x in tag[1]]

        fndElements = []

        if len(entTags):

            # Collect all the surface ids from phyiscal volume tags
            for entTag in entTags:
                els = self.getElements(entTag)
                fndElements.append(els)

        return self._mergeElements(fndElements)

    def getNodesFromEntity(self, entityId: Tuple[int,int]) -> np.array:
        """
        Returns all node ids from a selected entity in the GMSH model.

        :param entityId: The selected geometric entity id
        :return: The node ids for the selected entity
        """

        self.setAsCurrentModel()

        if not self._isMeshGenerated:
            raise Exception('Mesh is not generated')

        nodeIds = gmsh.model.mesh.getNodes(entityId[0], entityId[1], True)[0]
        nodeIds = np.sort(nodeIds)

        return nodeIds

    def getNodesByEntityName(self, entityName: str) -> np.array:
        """
        Returns all nodes for a selected surface region

        :param entityName: The geometric surface name
        :return: The list of node ids associated with the selected entity id
        """
        self.setAsCurrentModel()

        if not self._isMeshGenerated:
            raise Exception('Mesh is not generated')

        tagId = self.getIdByEntityName(entityName)

        nodeIds = gmsh.model.mesh.getNodes(tagId[0], tagId[1], True)[0]
        nodeIds = np.sort(nodeIds)

        return nodeIds

    def getNodesFromVolumeByName(self, volumeName: str) -> np.array:
        """
        Returns all node ids from a selected volume domain in the GMSH model.

        :param volumeName: Volume name
        :return: The list of node ids associated with the selected entity id
        """

        self.setAsCurrentModel()

        if not self._isMeshGenerated:
            raise Exception('Mesh is not generated')

        volTagId = self.getIdByVolumeName(volumeName)

        nodeIds = gmsh.model.mesh.getNodes(Ent.Volume, volTagId, True)[0]
        nodeIds = np.sort(nodeIds)

        return nodeIds

    def getNodesFromEdgeByName(self, edgeName: str):
        """
        Returns all nodes from a geometric edge

        :param edgeName: The geometric edge name
        :return: The list of node ids associated with the selected entity id
        """
        self.setAsCurrentModel()

        if not self._isMeshGenerated:
            raise Exception('Mesh is not generated')

        edgeTagId = self.getIdByEdgeName(edgeName)

        return gmsh.model.mesh.getNodes(Ent.Curve, edgeTagId, True)[0]

    def getNodesFromSurfaceByName(self, surfaceRegionName: str):
        """
        Returns all nodes for a selected surface region

        :param surfaceRegionName: The geometric surface name
        :return: The list of node ids associated with the selected entity id
        """
        self.setAsCurrentModel()

        if not self._isMeshGenerated:
            raise Exception('Mesh is not generated')

        surfTagId = self.getIdBySurfaceName(surfaceRegionName)

        return gmsh.model.mesh.getNodes(Ent.Surface, surfTagId, True)[0]

    def getSurfaceFacesFromRegion(self, regionName):

        self.setAsCurrentModel()

        surfTagId = self.getIdBySurfaceName(regionName)

        return self.getSurfaceFacesFromSurfId(surfTagId)

    def _getFaceOrderMask(self, elementType):
        """
        Private method which constructs the face mask array from the faces ordering of an element.
        :param elementType:
        :return:
        """
        mask = np.zeros([len(elementType.faces), np.max(elementType.faces)])
        for i in np.arange(mask.shape[0]):
            mask[i, np.array(elementType.faces[i]) - 1] = 1

        return mask

    def getFacesFromId(self, surfIds):

        self.setAsCurrentModel()

        if not isinstance(surfIds, list):
            surfId = [surfIds]

        faceNodeList = {3:[], 4:[]}
        for surfId in surfIds:
            modelEls = self.getElements((Ent.Surface, surfId))
            for elType in modelEls[0]:

                gmsh.model.mesh.getElementProperties(16)

                faceNodes = gmsh.model.mesh.getElementFaceNodes(elType, 3, surfId, primary=True).reshape(-1, 3)
                faceNodeList[3].append(faceNodes)
                faceNodes = gmsh.model.mesh.getElementFaceNodes(elType, 4, surfId, primary=True).reshape(-1, 4)
                faceNodeList[4].append(faceNodes)


    def writeMeshInput(self):
        """
        Generates the current mesh format as an abaqus (cal2culix) .inp representation format

        :return: THe mesh string format
        """
        # Print the nodes

        self.setAsCurrentModel()

        txt   = '*Heading\n'
        txt += 'mesh.inp\n'
        txt += '*node\n'

        nodeVals = gmsh.model.mesh.getNodes()
        nids = nodeVals[0]
        invNid = np.argsort(nids)
        nodeCoords = nodeVals[1].reshape(-1, 3)
        nodeCoords = nodeCoords[invNid]
        nids = np.sort(nids)

        # For any coordinate values that are below a epsilon, set these to zero

        for nid, nCoords in zip(nids, nodeCoords):
            ncoords = nodeCoords[int(nid) - 1]

            # Truncate any values below the epsilon
            ncoords[np.abs(ncoords) < self.EPSILON] = 0.0

            txt += "{:d}, {:e}, {:e}, {:e}\n".format(nid, *ncoords)

        # Obtain the surface-element physical groups and their associative element ids
        surfPhysicalGrps = gmsh.model.getPhysicalGroups(Ent.Surface)
        surfEntTags = [gmsh.model.getEntitiesForPhysicalGroup(x[0], x[1]) for x in surfPhysicalGrps]

        # Obtain the volume-element physical groups and their associative element ids
        volPhysicalGrps = gmsh.model.getPhysicalGroups(Ent.Volume)
        volEntTags = [gmsh.model.getEntitiesForPhysicalGroup(x[0], x[1]) for x in volPhysicalGrps]

        # Construct a type lookup index across all elements
        typeIdx = {}
        elNodeIdx = {}

        self.getNodesFromEntity((Ent.Surface, Ent.All))

        if len(volEntTags):

            # Collect all the surface ids from phyiscal volume tags
            for volEntTag in np.hstack(volEntTags):
                volEls = self.getElements((Ent.Volume, volEntTag))

                for elTyp, idx, nodes in zip(volEls[0], volEls[1], volEls[2]):
                    nodes = nodes.reshape(len(idx), -1)
                    for elId, elNodes in zip(idx, nodes):
                        typeIdx[elId] = elTyp
                        elNodeIdx[elId] = elNodes

        if len(surfEntTags):

            # Collect all the  ids from phyiscal surface tags
            for surfEntTag in np.hstack(surfEntTags):
                surfEls = self.getElements((Ent.Surface, surfEntTag))

                for elTyp, idx, nodes in zip(surfEls[0], surfEls[1], surfEls[2]):
                    nodes = nodes.reshape(len(idx), -1)
                    for elId, elNodes in zip(idx, nodes):
                        typeIdx[elId] = elTyp
                        elNodeIdx[elId] = elNodes

        txt += "******* E L E M E N T S ************* \n"
        for elTyp, elIds in self._meshAssignments.items():

            txt += "*ELEMENT, TYPE = {:s}\n".format(elTyp.name)

            for elId in elIds:
                if elTyp.id != typeIdx[elId]:
                    raise Exception('Incompatible elements selected (id: {:d} - {:s})'.format(elId, elTyp.name))

                # Sort element ids between gmsh and calculix abaqus
                elRow = elNodeIdx[elId]
                elSortRow = elRow[np.array(elTyp.map)-1].ravel()
                elLine = np.hstack([elId, elSortRow]).astype(np.int64)
                txt += np.array2string(elLine, precision=0, separator=', ', threshold=9999999999)[1:-1] + "\n"
                #txt +=  + str(elNodeIdx[elId])[1:-1] + "\n"

        return txt

    def getNodesInBoundingBox(self, minX: int, minY: int, minZ: int,
                                                             maxX: int, maxY: int, maxZ: int) -> np.array:
        """
        Returns all nodes within a bounding box

        :param minX: Minimum X
        :param minY: Minimum Y
        :param minZ: Minimum Z
        :param maxX: Maximum X
        :param maxY: Maximum Y
        :param maxZ: Maximum Z
        :return: The node ids within the bounding box
        """
        self.setAsCurrentModel()

        if not self._isMeshGenerated:
            raise Exception('Mesh is not generated')

        nodeVals = gmsh.model.mesh.getNodes()
        nids = nodeVals[0]
        invNid = np.argsort(nids)
        nodeCoords = nodeVals[1].reshape(-1, 3)
        nodeCoords = nodeCoords[invNid]
        nids = np.sort(nids)

        mask = np.logical_and(np.all(nodeCoords >= [minX, minY, minZ], axis=1),
                                              np.all(nodeCoords <= [maxX, maxY, maxZ], axis=1))

        return nids[mask]

    def getSurfaceFacesFromSurfId(self, surfTagId):

        self.setAsCurrentModel()

        if not self._isMeshGenerated:
            raise Exception('Mesh is not generated')

        mesh = gmsh.model.mesh

        surfNodeList2 = mesh.getNodes(Ent.Surface, surfTagId, True)[0]

        #surfNodeList2 = mesh.getNodesForEntity(2, surfTagId)[0]

        # Get tet elements
        tet4ElList = mesh.getElementsByType(elements.TET4.id)
        tet10ElList = mesh.getElementsByType(elements.TET10.id)

        tet4Nodes = tet4ElList[1].reshape(-1, 4)
        tet10Nodes = tet10ElList[1].reshape(-1, 10)

        tetNodes = np.vstack([tet4Nodes,
                              tet10Nodes[:, :4]])

        # Note subtract 1 to get an index starting from zero
        tetElList = np.hstack([tet4ElList[0]  -1,
                               tet10ElList[0] -1])

        print(elements.TET4.id)
        tetMinEl = np.min(tetElList)

        mask = np.isin(tetNodes, surfNodeList2)  # Mark nodes which are on boundary
        ab = np.sum(mask, axis=1)  # Count how many nodes were marked for each element
        fndIdx = np.argwhere(ab > 2)  # For all tets locate where the number of nodes on the surface = 3

        # Elements which belong onto the surface
        elIdx = tetElList[fndIdx]

        # Below if more than a four nodes (tet) lies on the surface, an error has occured
        if np.sum(ab > 3) > 0:
            raise ValueError('Instance of all nodes of tet where found')


        # Tet elements for Film [masks]

        surfFaces = np.zeros((len(elIdx), 2), dtype=np.uint32)
        surfFaces[:, 0] = elIdx.flatten()

        fMask = self._getFaceOrderMask(elements.TET4)

        # Iterate across each face of the element and apply mask across the elements
        for i in np.arange(fMask.shape[0]):
            surfFaces[mask[fndIdx.ravel()].dot(fMask[i]) == 3, 1] = 1  # Mask

        if False:
            F1 = [1, 1, 1, 0]  # 1: 1 2 3 = [1,1,1,0]
            F2 = [1, 1, 0, 1]  # 2: 1 4 2 = [1,1,0,1]
            F3 = [0, 1, 1, 1]  # 3: 2 4 3 = [0,1,1,1]
            F4 = [1, 0, 1, 1]  # 4: 3 4 1 = [1,0,1,1]

            surfFaces[mask[fndIdx.ravel()].dot(F1) == 3, 1] = 1  # Mask
            surfFaces[mask[fndIdx.ravel()].dot(F2) == 3, 1] = 2  # Mask
            surfFaces[mask[fndIdx.ravel()].dot(F3) == 3, 1] = 3  # Mask
            surfFaces[mask[fndIdx.ravel()].dot(F4) == 3, 1] = 4  # Mask

        # sort by faces
        surfFaces = surfFaces[surfFaces[:, 1].argsort()]

        surfFaces[:, 0] = surfFaces[:, 0] - (tetMinEl + 1)

        return surfFaces

    def renumberNodes(self) -> None:
        """
        Renumbers the nodes of the entire GMSH Model
        """
        self.setAsCurrentModel()

        if not self._isMeshGenerated:
            raise Exception('Mesh is not generated')

        gmsh.model.mesh.renumberNodes()
        self.setModelChanged()

    def renumberElements(self) -> None:
        """
        Renumbers the elements of the entire GMSH Model
        """
        self.setAsCurrentModel()

        if not self._isMeshGenerated:
            raise Exception('Mesh is not generated')

        gmsh.model.mesh.renumberElements()
        self.setModelChanged()

    def _setModelOptions(self) -> None:
        """
        Private method for initialising any additional options for individual models within GMSH which are not global.
        """
        self.setAsCurrentModel()


    def generateMesh(self, generate3DMesh = True) -> None:
        """
        Initialises the GMSH Meshing Procedure

        :param generate3DMesh: Generates the 3D Mesh -
        """
        self._isMeshGenerated = False

        self._setModelOptions()

        self.setAsCurrentModel()

        logging.info('Generating GMSH \n')

        gmsh.model.mesh.generate(Ent.Curve)

        gmsh.option.setNumber("Mesh.Algorithm", self._meshingAlgorithm2D)
        gmsh.model.mesh.generate(Ent.Surface)

        if generate3DMesh:

            try:

                gmsh.option.setNumber("Mesh.Algorithm", self._meshingAlgorithm3D)
                gmsh.model.mesh.generate(Ent.Volume)

            except:
                logging.error('Meshing Failed \n')

        self._isMeshGenerated = True
        self._isDirty = False

        elTypeIds = self.getElementTypes()
        self._meshAssignments = {}

    def getElementTypes(self):

        self.setAsCurrentModel()

        if not self._isMeshGenerated:
            logging.error('Mesh is not generated')
            return {}

        eType, elIds, nIds = gmsh.model.mesh.getElements()
        elTypeIds = {} #np.zeros_like(np.hstack(elIds))

        for et, eids in zip(eType, elIds):
            for eid in eids:
                elTypeIds[int(eid)-1] = et

        return elTypeIds

    def getNumberElements(self, physicalGroupsOnly = True):
        """
        The number of elements in the mesh

        :param physicalGroupsOnly: If `True`` mesh number consists only physical grups - default is `True`
        :return:
        """
        self.setAsCurrentModel()

        if not self._isMeshGenerated:
            logging.error('Mesh is not generated')
            return

        if physicalGroupsOnly:
            return len(self.getAllPhysicalGroupElements())
        else:

            eType, elIds, nIds = gmsh.model.mesh.getElements()
            return len(np.hstack(np.hstack(elIds)).ravel())

    def setRecombineSurfaces(self, surfId, angle = 45):
        """
        Set a constraints on the surface mesh (triangles) to be recombined to quadrilaterals based on the angle

        :param surfId: The surface identity
        :param angle: The angle in degrees for recombination
        """

        self.setAsCurrentModel()

        if not self._isMeshGenerated:
            logging.error('Mesh is not generated')
            return

        gmsh.model.mesh.setRecombine(Ent.Surface, surfId, angle)

    def recombineMesh(self):
        """
        Recombines the surface mesh - between triangles and quadrilaterals using the specified recombination algorithm.
        """
        self.setAsCurrentModel()

        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", self._recombinationAlgorithm)

        if not self._isMeshGenerated:
            logging.error('Mesh is not generated')
            return

        logging.info('Recombining Mesh GMSH \n')

        try:
            gmsh.model.mesh.recombine()
        except:
            logging.error('Recombining Failed \n')

        elTypeIds = self.getElementTypes()
        # for val in elTypeIds.items():
        #     eAssignments[val[0]] = 0

        self._meshAssignments = {}

        self._isMeshGenerated = True

    def isMeshGenerated(self) -> bool:
        """
        Returns 'True' if the mesh has been successfully internally generated by GMSH
        """
        return self._isMeshGenerated

    def writeMesh(self, filename: str) -> None:
        """
        Writes the generated mesh to the file

        :param filename: str - Filename (including the type) to save to.
        """

        if self.isMeshGenerated():
            self.setAsCurrentModel()

            with open(filename, 'w') as f:

                out = self.writeMeshInput()
                f.write(out)
        else:
            raise ValueError('Mesh has not been generated before writing the file')

    def getIdByEntityName(self, entityName: str) -> int:
        """
        Obtains the ID for volume name

        :param entityName: str
        :return: int: Volume ID
        """
        self.setAsCurrentModel()

        vols = gmsh.model.getEntities()
        names = [(gmsh.model.getEntityName(x[0], x[1]), x) for x in vols]

        tagId = -1
        for name in names:
            if name[0] == entityName:
                tagId = name[1]

        if tagId == -1:
            raise ValueError('Volume region ({:s}) was not found'.format(entityName))

        return tagId


    def getIdByVolumeName(self, volumeName: str) -> int:
        """
        Obtains the ID for volume name

        :param volumeName: str
        :return: int: Volume ID
        """
        self.setAsCurrentModel()

        vols = gmsh.model.getEntities(Ent.Volume)
        names = [(gmsh.model.getEntityName(Ent.Volume, x[1]), x[1]) for x in vols]

        volTagId = -1
        for name in names:
            if name[0] == volumeName:
                volTagId = name[1]

        if volTagId == -1:
            raise ValueError('Volume region ({:s}) was not found'.format(volumeName))

        return volTagId

    def getIdByEdgeName(self, edgeName: str) -> int:
        """
        Obtains the ID for the edge name

        :param edgeName: Geometric edge name
        :return: Edge ID
        """

        self.setAsCurrentModel()

        surfs = gmsh.model.getEntities(Ent.Curve)
        names = [(gmsh.model.getEntityName(Ent.Curve, x[1]), x[1]) for x in surfs]

        edgeTagId = -1
        for name in names:
            if name[0] == edgeName:
                surfTagId = name[1]

        if edgeTagId == -1:
            raise ValueError('Surface region ({:s}) was not found'.format(edgeName))

        return edgeTagId

    def getIdBySurfaceName(self, surfaceName : str) -> int:
        """
        Obtains the ID for the surface name

        :param surfaceName:  Geometric surface name
        :return: Surface Ids
        """

        self.setAsCurrentModel()

        surfs = gmsh.model.getEntities(Ent.Surface)
        names = [(gmsh.model.getEntityName(Ent.Surface, x[1]), x[1]) for x in surfs]

        surfTagId = -1
        for name in names:
            if name[0] == surfaceName:
                surfTagId = name[1]

        if surfTagId == -1:
            raise ValueError('Surface region ({:s}) was not found'.format(surfaceName))

        return surfTagId

    def setEdgeSet(self, grpTag, edgeName):
        # Adding a physical group will export the surface mesh later so these need removing

        self.edgeSets.append({'name': edgeName, 'tag': grpTag, 'nodes': np.array([])})

        # below is not needed - it is safe to get node list directly from entity

    #        gmsh.model.addPhysicalGroup(1, [grpTag], len(self.edgeSets))
    #        gmsh.model.setPhysicalName(1, len(self.edgeSets), edgeName)

    def setSurfaceSet(self, grpTag: int, surfName:str) -> None:
        """
        Generates a surface set based on the geometric surface name. GMSH creates an associative surface mesh, which will
        later be automatically removed.

        :param grpTag: int: A unique Geometric Id used to associate the surface set as a physical group
        :param surfName: str: The surface name for the set
        :return:
        """
        # Adding a physical group will export the surface mesh later so these need removing

        self.setAsCurrentModel()
        self.surfaceSets.append({'name': surfName, 'tag': grpTag, 'nodes': np.array([])})

        gmsh.model.addPhysicalGroup(Ent.Surface, [grpTag], len(self.surfaceSets))
        gmsh.model.setPhysicalName(Ent.Surface, len(self.surfaceSets), surfName)
        gmsh.model.removePhysicalGroups()
        self.setModelChanged()

    def interpTri(triVerts, triInd):

        from matplotlib import pyplot as plt

        import matplotlib.pyplot as plt
        import matplotlib.tri as mtri

        x = triVerts[:, 0]
        y = triVerts[:, 1]
        triang = mtri.Triangulation(triVerts[:, 0], triVerts[:, 1], triInd)

        # Interpolate to regularly-spaced quad grid.
        z = np.cos(1.5 * x) * np.cos(1.5 * y)
        xi, yi = np.meshgrid(np.linspace(0, 3, 20), np.linspace(0, 3, 20))

        interp_lin = mtri.LinearTriInterpolator(triang, z)
        zi_lin = interp_lin(xi, yi)

        interp_cubic_geom = mtri.CubicTriInterpolator(triang, z, kind='geom')
        zi_cubic_geom = interp_cubic_geom(xi, yi)

        interp_cubic_min_E = mtri.CubicTriInterpolator(triang, z, kind='min_E')
        zi_cubic_min_E = interp_cubic_min_E(xi, yi)

        # Set up the figure
        fig, axs = plt.subplots(nrows=2, ncols=2)
        axs = axs.flatten()

        # Plot the triangulation.
        axs[0].tricontourf(triang, z)
        axs[0].triplot(triang, 'ko-')
        axs[0].set_title('Triangular grid')

        # Plot linear interpolation to quad grid.
        axs[1].contourf(xi, yi, zi_lin)
        axs[1].plot(xi, yi, 'k-', lw=0.5, alpha=0.5)
        axs[1].plot(xi.T, yi.T, 'k-', lw=0.5, alpha=0.5)
        axs[1].set_title("Linear interpolation")

        # Plot cubic interpolation to quad grid, kind=geom
        axs[2].contourf(xi, yi, zi_cubic_geom)
        axs[2].plot(xi, yi, 'k-', lw=0.5, alpha=0.5)
        axs[2].plot(xi.T, yi.T, 'k-', lw=0.5, alpha=0.5)
        axs[2].set_title("Cubic interpolation,\nkind='geom'")

        # Plot cubic interpolation to quad grid, kind=min_E
        axs[3].contourf(xi, yi, zi_cubic_min_E)
        axs[3].plot(xi, yi, 'k-', lw=0.5, alpha=0.5)
        axs[3].plot(xi.T, yi.T, 'k-', lw=0.5, alpha=0.5)
        axs[3].set_title("Cubic interpolation,\nkind='min_E'")

        fig.tight_layout()
        plt.show()