import logging
import os
import numpy as np
import gmsh
from enum import Enum
from typing import List, Optional, Tuple


class MeshingAlgorithm(Enum):
    DELAUNAY = 1
    FRONTAL = 4
    FRONTAL_DELAUNAY = 5
    FRONTAL_HEX = 6
    MMG3D = 7
    RTREE = 9
    HXT = 10


class ElementType:
    """
    Element types information used via GMSH and Calculix
    """

    class NODE:
        """ A single node element"""
        id = 15
        name = 'Node'
        nodes = 1
        faces = None

    class TET4:
        """ 1st order linear Tet Element (C3D4) """
        id = 4
        name =  'C3D4'
        nodes = 4
        faces = np.array([[1,2,3], [1,4,2], [2,4,3], [3,4,1]])

    class TET10:
        """ 2nd order Quadratic Tet Element (C3D10) consisting of 10 nodes """
        id = 11
        name = 'C3D10'
        nodes = 4
        faces = np.array([[1,2,3], [1,4,2], [2,4,3], [3,4,1]])

    class HEX8:
        """ Linear Hex Element (C3D8) """
        id = 5
        name = 'C3D8'
        nodes = 8
        faces = np.array([[1,2,3,4], [5,8,7,6], [1,5,6,2], [2,6,7,3], [3,7,8,4], [4,8,5,1]])

    class HEX8R:
        """Linear Hex Element (C3D8R) with reduced order integration """
        id = 5
        name = 'C3D8R'
        nodes = 8
        faces = np.array([[1, 2, 3, 4], [5, 8, 7, 6], [1, 5, 6, 2], [2, 6, 7, 3], [3, 7, 8, 4], [4, 8, 5, 1]])

    class HEX8R:
        """
        Linear Hex Element (C3D8I) with reformulation to reduce the effects of shear and
        volumetric locking and hourglass effects under some extreme situations
        """
        id = 5
        name = 'C3D8I'
        nodes = 8
        faces = np.array([[1, 2, 3, 4], [5, 8, 7, 6], [1, 5, 6, 2], [2, 6, 7, 3], [3, 7, 8, 4], [4, 8, 5, 1]])

    class HEX8R:
        """
        Quadratic Hex Element (C3D20) consisting of 20 Nodes
        """
        id = 17
        name = 'C3D20'
        nodes = 20
        faces = np.array([[1, 2, 3, 4], [5, 8, 7, 6], [1, 5, 6, 2], [2, 6, 7, 3], [3, 7, 8, 4], [4, 8, 5, 1]])


    class WEDGE6:
        """ Wedge or Prism Element (C3D6) """
        id = 6
        name = 'C3D6'
        nodes = 6
        faces = np.array([[1,2,3], [4,5,6], [1,2,5,4], [2,3,6,5], [3,1,4,6]])



class Mesher:
    """
    The Mesher class provides the base interface built upon the GMSH-SDK API operations. It provides the capability
    to mesh multiple PythonOCC objects
    """

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

        self._isMeshGenerated = False
        self._isDirty = False # Flag to indicate model hasn't been generated and is dirty
        self._isGeometryDirty = False
        self._meshingAlgorithm = MeshingAlgorithm.DELAUNAY

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

    def maxPhysicalGroupId(self, dim: int) -> int:
        """
        Returns the highest physical group id in the GMSH model

        :param dim: int: The chosen dimension
        :return: int: The highest group id used
        """
        self.setAsCurrentModel()
        return np.max([x[1] for x in gmsh.model.getPhysicalGroups(dim)])

    def getVolumeName(self, volId: int) -> str:
        """
        Gets the volume name (if assigned)

        :param volId: int: Volume id of a region
        """
        self.setAsCurrentModel()
        return gmsh.model.getEntityName(3,volId)

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

    def setVolumePhysicalName(self, volId: int, name: str) -> None:
        """
        Sets the geometric name of the volume id

        :param volId: Volume Id
        :param name: Name assigned to volume
        """
        self.setAsCurrentModel()
        maxId = self.maxPhysicalGroupId(3)
        gmsh.model.addPhysicalGroup(3, [volId],maxId+1)
        gmsh.model.setPhysicalName(3,volId, name)

    def setSurfacePhysicalName(self, surfId: int, name: str) -> None:
        """
        Sets the geometric name of the surface id

        :param surfId: Volume Id
        :param name: Name assigned to volume
        """
        self.setAsCurrentModel()
        maxId = self.maxPhysicalGroupId(2)
        gmsh.model.addPhysicalGroup(2, [surfId],maxId+1)
        gmsh.model.setPhysicalName(2,surfId, name)

    def setEntityPhysicalName(self, id, name: str) -> None:
        """
        Sets the geometric name of the volume id

        :param id: tuple(int, int): Dimension, Entity Id
        :param name: str: Name assigned to entity
        """
        self.setAsCurrentModel()
        maxId = self.maxPhysicalGroupId(id[0])
        gmsh.model.addPhysicalGroup(id[0], [id[1]],maxId+1)
        gmsh.model.setPhysicalName(id[0], id[1], name)

    def name(self) -> str:
        """
        Each GMSH model requires a name
        """
        return self._modelName

    def isDirty(self) -> bool:
        """
        Has the model been successfully generated and no pending modifications exist.
        """

        return self._isDirty

    def setModelChanged(self, state : bool = False) -> None:
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

        gmsh.model.setEntityName(3, len(self.geoms), name)
        gmsh.model.addPhysicalGroup(3, [len(self.geoms)], len(self.geoms))
        gmsh.model.setPhysicalName(3, len(self.geoms), name)

        # set the mesh size for this geometry
        bbox = self.getGeomBoundingBoxById(len(self.geoms))
        extents = bbox[1, :] - bbox[0, :]
        avgDim = np.mean(extents)
        meshSize = avgDim * meshFactor

        print('GMSH: Avg dim', avgDim, ' mesh size: ', meshSize)
        geomPoints = self.getPointsFromVolume(len(self.geoms))

        # Set the geometry volume size
        self.geoms[-1]['meshSize'] = meshSize

        self.setMeshSize(geomPoints, meshSize)

        self._isGeometryDirty = True
        self.setModelChanged()

    def clearMesh(self) -> None:
        """
        Clears any meshes previously generated
        """
        self.setAsCurrentModel()
        gmsh.model.mesh.clear()

    @property
    def volumes(self) -> List[int]:
        self.setAsCurrentModel()
        tags = gmsh.model.getEntities(3)
        return [(x[1]) for x in tags]

    @property
    def surfaces(self) -> List[int]:
        """
        Returns all the surface ids of geometry in the model
        """
        self.setAsCurrentModel()
        tags = gmsh.model.getEntities(2)
        return [(x[1]) for x in tags]

    @property
    def edges(self) -> List[int]:
        tags = gmsh.model.getEntities(1)
        return [(x[1]) for x in tags]

    @property
    def points(self) -> List[int]:
        """
        Returns all the point ids
        :return:
        """
        self.setAsCurrentModel()
        tags = gmsh.model.getEntities(0)
        return [(x[1]) for x in tags]

    def mergeGeometry(self) -> None:
        """
        Geometry is merged/fused together. Coincident surfaces and points are
        automatically merged together which enables a coherent mesh to be
        generated
        """

        self.setAsCurrentModel()

        volTags = gmsh.model.getEntities(3)
        out = gmsh.model.occ.fragment(volTags[1:], [volTags[0]])

        # Synchronise the model
        gmsh.model.occ.synchronize()

        self._isGeometryDirty = True
        self.setModelChanged()

    def boundingBox(self):
        self.setAsCurrentModel()
        return np.array(gmsh.model.getBoundingBox())

    def getGeomBoundingBoxById(self, tagId: int):
        self.setAsCurrentModel()
        return np.array(gmsh.model.getBoundingBox(3, tagId)).reshape(-1, 3)

    def getGeomBoundingBoxByName(self, volumeName):
        self.setAsCurrentModel()
        volTagId = self.getIdByVolumeName(volumeName)
        return np.array(gmsh.model.getBoundingBox(3, volTagId)).reshape(-1, 3)

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

    def setMeshingAlgorithm(self, meshingAlgorithm) -> None:
        """
        Sets the meshing algorithm to use by GMSH for the model
        
        :param meshingAlgorithm: MeshingAlgorith
        """

        # The meshing algorithm is applied before generation, as this may be model specific
        print(meshingAlgorithm)
        self._meshingAlgorithm = meshingAlgorithm
        self.setModelChanged()

    ## Class Methods
    @classmethod
    def setUnits(cls, unitVal):
        cls.Units = unitVal

        if cls.Initialised:
            gmsh.option.setString("Geometry.OCCTargetUnit", Mesher.Units);

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

        print('Initialising GMSH \n')

        gmsh.initialize()

        # Mesh.Algorithm3D
        # 3D mesh algorithm (1: Delaunay, 4: Frontal, 5: Frontal Delaunay, 6: Frontal Hex, 7: MMG3D, 9: R-tree, 10: HXT)
        # Default value: 1#

        gmsh.option.setNumber("Mesh.Algorithm", MeshingAlgorithm.FRONTAL_DELAUNAY.value);
        #        gmsh.option.setNumber("Mesh.Algorithm3D", 10);

        gmsh.option.setNumber("Mesh.ElementOrder", Mesher.ElementOrder)
        #        gmsh.option.setNumber("Mesh.OptimizeNetgen",1)
        gmsh.option.setNumber("Mesh.MaxNumThreads3D", Mesher.NumThreads)

        # gmsh.option.setNumber("Mesh.SaveGroupsOfNodes", 1);
        # gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 15);

        gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0);
        gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 1);
        gmsh.option.setNumber("Mesh.SaveAll", 0);

        # OCC Options
        gmsh.option.setNumber("Geometry.OCCFixDegenerated", 1);
        gmsh.option.setString("Geometry.OCCTargetUnit", Mesher.Units);

        # General Options
        gmsh.option.setString("General.ErrorFileName", 'error.log');
        gmsh.option.setNumber("General.Terminal", 1)

        ########## Gui Options ############
        gmsh.option.setNumber("General.Antialiasing", 1)
        gmsh.option.setNumber("Geometry.SurfaceType", 2)

        # The following GMSH options do not appear to change anything
        # Import labels is required inorder to correctly reference geometric entities and their associated mesh entities
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

        tags = gmsh.model.getPhysicalGroups(1)

        for tag in tags:
            # Remove all tri group surfaces
            print('removing edge {:s}'.format(gmsh.model.getPhysicalName(1, tag[1])))
            gmsh.model.removePhysicalGroups(tag)

        self.setModelChanged()

    def removeSurfaceMeshes(self):
        """
        Removes surface meshes (2D mesh entities) from the GMSH model
        """
        self.setAsCurrentModel()

        tags = gmsh.model.getPhysicalGroups(2)

        for tag in tags:
            # Remove all tri group surfaces
            print('removing surface {:s}'.format(gmsh.model.getPhysicalName(2, tag[1])))
            gmsh.model.removePhysicalGroups(tag)

        self.setModelChanged()

    # Geometric methods
    def getPointsFromVolume(self, id: int) -> List[int]:
        """
        From a Volume Id, obtain all Point Ids associated with this volume - note may include shared points.

        :param id: Volume ID
        :return: list(int) - List of Point Ids
        """
        self.setAsCurrentModel()
        pnts = gmsh.model.getBoundary([(3, id)], recursive=True)
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
        surfs = gmsh.model.getBoundary( [(3, id)], recursive=False)
        return [x[1] for x in surfs]

    def getPointsFromVolumeByName(self, volumeName: str):
        """
        Returns all geometric points from a given volume domain by its name (if assigned)
        :param volumeName: volumeName
        :return:
        """
        return self.getPointsFromVolume(self.getIdBySurfaceName(volumeName))

    # Mesh methods
    def getNodeIds(self):
        """
        Returns the nodal ids from the entire GMSH model
        :return:
        """
        self.setAsCurrentModel()

        if not self._isMeshGenerated:
            raise ValueError('Mesh is not generated')

        nodeList = gmsh.model.mesh.getNodes()
        return nodeList[0]

    def getNodes(self):
        """
        Returns the nodal coordinates from the entire GMSH model

        :return:
        """
        self.setAsCurrentModel()


        if not self._isMeshGenerated:
            raise ValueError('Mesh is not generated')

        nodeList = gmsh.model.mesh.getNodes()

        nodeCoords = nodeList[1].reshape(-1, 3)

        nodeCoordsSrt = nodeCoords[np.sort(nodeList[0]) - 1]
        return nodeCoordsSrt  # , np.sort(nodeList[0])-1


    def getElements(self, entityId: Tuple[int,int] = None):
        """
        Returns the elements for the entire model or optionally a specific entity.
        :param entityId: The entity id to obtain elements for
        :return:
        """
        self.setAsCurrentModel()

        if not self._isMeshGenerated:
            raise ValueError('Mesh is not generated')

        if entityId:
            # return all the elements for the entity
            result =  gmsh.model.mesh.getElements(entityId[0], entityId[1])
            return np.hstack(result[1])
        else:
            # Return all the elements in the model
            result =  gmsh.model.mesh.getElements()
            return np.hstack(result[1])


    def getElementsByType(self, elType) -> np.ndarray:
        """
        Returns all elements of type (elType) from the GMSH model, within class ElementTypes. Note: the element ids are returned with
        an index starting from 1 - internally GMSH uses an index starting from 1, like most FEA pre-processors

        :return: List of element Ids.
        """

        self.setAsCurrentModel()

        if not self._isMeshGenerated:
            raise ValueError('Mesh is not generated')

        elVar = gmsh.model.mesh.getElementsByType(elType.id)
        elements = elVar[1].reshape(-1, elType.nodes) # TODO check if necessary - 1

        return elements


    def getNodesFromEntity(self, entityId: Tuple[int,int]) -> np.ndarray:
        """
        Returns all node ids from a selected entity in the GMSH model.

        :param entityId:  int : Volume name
        :return:
        """

        self.setAsCurrentModel()

        if not self._isMeshGenerated:
            raise ValueError('Mesh is not generated')

        return gmsh.model.mesh.getNodes(entityId[0], entityId[1], True)[0]

    def getNodesByEntityName(self, entityName: str) -> np.ndarray:
        """
        Returns all nodes for a selected surface region

        :param entityName: The geometric surface name
        :return: Node Ids
        """
        self.setAsCurrentModel()

        if not self._isMeshGenerated:
            raise ValueError('Mesh is not generated')

        tagId = self.getIdByEntityName(entityName)

        return gmsh.model.mesh.getNodes(tagId[0], tagId[1], True)[0]

    def getNodesFromVolumeByName(self, volumeName: str):
        """
        Returns all node ids from a selected volume domain in the GMSH model.

        :param volumeName:  Volume name
        :return:
        """

        self.setAsCurrentModel()

        if not self._isMeshGenerated:
            raise ValueError('Mesh is not generated')

        volTagId = self.getIdByVolumeName(volumeName)

        return gmsh.model.mesh.getNodes(3, volTagId, True)[0]

    def getNodesFromEdgeByName(self, edgeName: str):
        """
        Returns all nodes from a geometric edge

        :param edgeName: The geometric edge name
        :return:
        """
        self.setAsCurrentModel()

        if not self._isMeshGenerated:
            raise ValueError('Mesh is not generated')

        edgeTagId = self.getIdByEdgeName(edgeName)

        return gmsh.model.mesh.getNodes(1, edgeTagId, True)[0]


    def getNodesFromSurfaceByName(self, surfaceRegionName: str):
        """
        Returns all nodes for a selected surface region

        :param surfaceRegionName: The geometric surface name
        :return:
        """
        self.setAsCurrentModel()

        if not self._isMeshGenerated:
            raise ValueError('Mesh is not generated')

        surfTagId = self.getIdBySurfaceName(surfaceRegionName)

        return gmsh.model.mesh.getNodes(2, surfTagId, True)[0]


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
        mask = np.zeros([elementType.faces.shape[0], np.max(elementType.faces)])
        for i in np.arange(mask.shape[0]):
            mask[i, elementType.faces[i] - 1] = 1

        return mask


    def getSurfaceFacesFromSurfId(self, surfTagId):

        self.setAsCurrentModel()

        if not self._isMeshGenerated:
            raise ValueError('Mesh is not generated')


        mesh = gmsh.model.mesh

        surfNodeList2 = mesh.getNodes(2, surfTagId, True)[0]

        #surfNodeList2 = mesh.getNodesForEntity(2, surfTagId)[0]

        # Get tet elements
        tet4ElList = mesh.getElementsByType(ElementType.TET4.id)
        tet10ElList = mesh.getElementsByType(ElementType.TET10.id)

        tet4Nodes = tet4ElList[1].reshape(-1, 4)
        tet10Nodes = tet10ElList[1].reshape(-1, 10)

        tetNodes = np.vstack([tet4Nodes,
                              tet10Nodes[:, :4]])

        # Note subtract 1 to get an index starting from zero
        tetElList = np.hstack([tet4ElList[0]  -1,
                               tet10ElList[0] -1])

        print(ElementType.TET4.id)
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

        fMask = self._getFaceOrderMask(ElementType.TET4)

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
            raise ValueError('Mesh is not generated')

        gmsh.model.mesh.renumberNodes()
        self.setModelChanged()

    def renumberElements(self) -> None:
        """
        Renumbers the elements of the entire GMSH Model
        """
        self.setAsCurrentModel()

        if not self._isMeshGenerated:
            raise ValueError('Mesh is not generated')

        gmsh.model.mesh.renumberElements()
        self.setModelChanged()

    def _setModelOptions(self) -> None:
        """
        Private method for initialising any additional options for individual models within GMSH which are not global.
        """
        self.setAsCurrentModel()
        gmsh.option.setNumber("Mesh.Algorithm", self._meshingAlgorithm.value)


    def generateMesh(self) -> None:
        """
        Initialises the GMSH Meshing Proceedure.
        """
        self._isMeshGenerated = False

        self._setModelOptions()

        self.setAsCurrentModel()

        print('Generating GMSH \n')

        gmsh.model.mesh.generate(1)
        gmsh.model.mesh.generate(2)

        try:
            gmsh.model.mesh.generate(3)
        except:
            print('Meshing Failed \n')

        self._isMeshGenerated = True
        self._isDirty = False

    def isMeshGenerated(self) -> bool:
        """
        Returns if the mesh has been successfully generated by GMSH
        """
        return self._isMeshGenerated

    def writeMesh(self, filename: str) -> None:
        """
        Writes the generated mesh to the file

        :param filename: str - Filename (including the type) to save to.
        """

        if self.isMeshGenerated():
            self.setAsCurrentModel()
            gmsh.write(filename)
        else:
            raise ValueError('Mesh has not been generated before writing the file')

    def getIdByEntityName(self, entityName: str) -> int:
        """
        Obtains the ID for volume name

        :param volumeName: str
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

        vols = gmsh.model.getEntities(3)
        names = [(gmsh.model.getEntityName(3, x[1]), x[1]) for x in vols]

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

        surfs = gmsh.model.getEntities(1)
        names = [(gmsh.model.getEntityName(1, x[1]), x[1]) for x in surfs]

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

        surfs = gmsh.model.getEntities(2)
        names = [(gmsh.model.getEntityName(2, x[1]), x[1]) for x in surfs]

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
        gmsh.model.mesh.get
        gmsh.model.addPhysicalGroup(2, [grpTag], len(self.surfaceSets))
        gmsh.model.setPhysicalName(2, len(self.surfaceSets), surfName)
        gmsh.model.removePhysicalGroups()
        self.setModelChanged()

    def interpTri(triVerts, triInd):

        from matplotlib import pyplot as plt

        import matplotlib.pyplot as plt
        import matplotlib.tri as mtri
        import numpy as np

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