import gmsh
import numpy as np
from .mesher import Mesher

def removeSurfaceMeshes(model: Mesher) -> None:
    """
    In order to assign face based boundary conditions to surfaces (e.g. flux, convection), the surface mesh is compared
    to the volumetric mesh to identify the actual surface mesh. This is then removed afterwards.

    :param model: Mesher: The GMSH  model
    """
    tags = model.getPhysicalGroups(2)

    for tag in tags:
        # Remove all tri group surfaces
        print('removing surface {:s}'.format(model.getPhysicalName(2, tag[1])))
        model.removePhysicalGroups(tag)

def getNodesFromVolume(volumeName : str, model: Mesher):
    """
    Gets the nodes for a specified volume

    :param volumeName: str - The volume domain in the model to obtain the nodes from
    :param model: Mesher: The GMSH  model
    :return:
    """
    vols = model.getPhysicalGroups(3)
    names = [(model.getPhysicalName(3, x[1]), x[1]) for x in vols]

    volTagId = -1
    for name in names:
        if name[0] == volumeName:
            volTagId = name[1]

    if volTagId == -1:
        raise ValueError('Volume region ({:s}) was not found'.format(volumeName))

    return model.mesh.getNodesForPhysicalGroup(3, volTagId)[0]

def getNodesFromRegion(surfaceRegionName: str, model : Mesher):
    """
    Gets the nodes for a specified surface region

    :param surfaceRegionName: str - The volume domain in the model to obtain the nodes from
    :param model: Mesher: The GMSH  model
    :return:
    """
    surfs = model.getPhysicalGroups(2)
    names = [(model.getPhysicalName(2, x[1]), x[1]) for x in surfs]

    surfTagId = -1
    for name in names:
        if name[0] == surfaceRegionName:
            surfTagId = name[1]

    if surfTagId == -1:
        raise ValueError('Surface region ({:s}) was not found'.format(surfaceRegionName))

    return model.mesh.getNodesForPhysicalGroup(2, surfTagId)[0]


def getSurfaceFacesFromRegion(regionName, model):
    """
     Gets the faces from a surface region, which are compatible with GMSH in order to apply surface BCs to.

     :param surfaceRegionName: str - The volume domain in the model to obtain the nodes from
     :param model: Mesher: The GMSH  model
     :return:
     """

    surfs = model.getPhysicalGroups(2)
    names = [(model.getPhysicalName(2, x[1]), x[1]) for x in surfs]

    surfTagId = -1
    for name in names:
        if name[0] == regionName:
            surfTagId = name[1]

    if surfTagId == -1:
        raise ValueError('Surface region ({:s}) was not found'.format(regionName))

    mesh = model.mesh

    surfNodeList2 = mesh.getNodesForPhysicalGroup(2, surfTagId)[0]

    # Get tet elements
    tetElList = mesh.getElementsByType(4)

    tetNodes = tetElList[1].reshape(-1, 4)
    tetMinEl = np.min(tetElList[0])

    mask = np.isin(tetNodes, surfNodeList2)  # Mark nodes which are on boundary
    ab = np.sum(mask, axis=1)  # Count how many nodes were marked for each element
    fndIdx = np.argwhere(ab > 2)  # For all tets
    elIdx = tetElList[0][fndIdx]

    if np.sum(ab > 3) > 0:
        raise ValueError('Instance of all nodes of tet where found')

    # Tet elements for Film [masks]
    F1 = [1, 1, 1, 0]  # 1: 1 2 3 = [1,1,1,0]
    F2 = [1, 1, 0, 1]  # 2: 1 4 2 = [1,1,0,1]
    F3 = [0, 1, 1, 1]  # 3: 2 4 3 = [0,1,1,1]
    F4 = [1, 0, 1, 1]  # 4: 3 4 1 = [1,0,1,1]

    surfFaces = np.zeros((len(elIdx), 2), dtype=np.uint32)
    surfFaces[:, 0] = elIdx.flatten()

    surfFaces[mask[fndIdx.ravel()].dot(F1) == 3, 1] = 1  # Mask
    surfFaces[mask[fndIdx.ravel()].dot(F2) == 3, 1] = 2  # Mask
    surfFaces[mask[fndIdx.ravel()].dot(F3) == 3, 1] = 3  # Mask
    surfFaces[mask[fndIdx.ravel()].dot(F4) == 3, 1] = 4  # Mask

    # sort by faces
    surfFaces = surfFaces[surfFaces[:, 1].argsort()]

    surfFaces[:, 0] = surfFaces[:, 0] - (tetMinEl + 1)

    return surfFaces