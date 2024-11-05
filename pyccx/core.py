import abc
from typing import Any, List, Optional, Tuple

import numpy as np

class ModelObject():

    def __init__(self, name, label = ''):

        self.setName(name)
        self._label = label

    @property
    def label(self) -> str:
        return self._label

    @label.setter
    def label(self, label: str):
        self._label = label

    @property
    def name(self) -> str:

        return self._name

    @name.setter
    def name(self, name):
        self.setName(name)

    def setName(self, name: str):

        if not name.isascii():
            raise ValueError('Name provided ({:s}) must be alpha-numeric'.format(name))

        if ' ' in name:
            raise ValueError('Name provided ({:s}) must not contain spaces'.format(name))

        if '*' in name:
            raise ValueError('Name provide ({:s}) contains invalid character (*)'.format(name))

        self._name = name

class Amplitude(ModelObject):

    def __init__(self, name: str, profile = None):

        super().__init__(name)

        self._profile = profile

    @property
    def profile(self):
        return self._profile

    @profile.setter
    def profile(self, profile):

        profile = np.asanyarray(profile)

        if not (profile.ndim == 2 and profile.shape[1] == 2):
            raise ValueError('Invalid profile passed to Amplitude')

        self._profile = profile

    def writeInput(self) -> str:

        out = '*AMPLITUDE, NAME={:s}\n'.format(self.name)

        for row in self.profile:
            time, amplitude = row
            out += '{:.5f}, {:.5f}\n'.format(time, amplitude)

        return out


class MeshSet:
    """
    The Mesh set is a basic entity for storing node and element set lists that are used for creating sets across
    both node and element types.
    """
    def __init__(self, name):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name):
        self._name = name


class NodeSet(MeshSet):
    """
     An node set is basic entity for storing node set lists. The set remains constant without any dynamic referencing
     to any underlying geometric entities.
     """
    def __init__(self, name, nodes):
        super().__init__(name)
        self._nodes = np.unique(np.asanyarray(nodes, dtype=np.int64))

    @property
    def nodes(self):
        """
        Nodes contains the list of Node IDs
        """
        return self._nodes

    @nodes.setter
    def nodes(self, nodes):
        self._nodes = np.unique(np.asanyarray(nodes, dtype=np.int64))

    def writeInput(self) -> str:
        out = '*NSET, NSET={:s}\n'.format(self.name)
        for i in range(0, self.nodes.shape[0], 16):
            out += ', '.join(['{0:6d}'.format(val) for val in self.nodes[i:i+16]])
            out += '\n'
        return out


class ElementSet(MeshSet):
    """
    An element set is basic entity for storing element set lists.The set remains constant without any dynamic referencing
    to any underlying geometric entities.
    """
    def __init__(self, name, els):
        super().__init__(name)
        self._els = np.unique(np.asanyarray(els))

    @property
    def els(self):
        """
        Elements contains the list of element IDs
        """
        return self._els

    @els.setter
    def els(self, elements: np.array):
        self._els = np.unique(np.asanyarray(elements, dtype=np.int64))

    def writeInput(self) -> str:

        out = '*ELSET, ELSET={:s}\n'.format(self.name)

        for i in range(0, self._els.shape[0], 16):
            out += ', '.join(['{0:6d}'.format(val) for val in self._els[i:i+16]])
            out += '\n'

        return out


class SurfaceNodeSet(MeshSet):
    """
    A surface-node set set is basic entity for storing element face lists, typically for setting directional fluxes onto
    surface elements based on the element ordering. The set remains constant without any dynamic referencing
    to any underlying geometric entities. This approach requires explicitly assigning the list of nodal ids that
    define the surface.
    """
    def __init__(self, name, nodalSet):

        super().__init__(name)
        self._surfaceNodes = np.asanyarray(nodalSet)

    @property
    def surfacePairs(self) -> np.array:
        """
        Elements with the associated face orientations are specified as Nx2 numpy array, with the first column being
        the element Id, and the second column the chosen face orientation
        """
        return self._elSurfacePairs

    @surfacePairs.setter
    def surfacePairs(self, surfacePairs):
        self._elSurfacePairs = np.asanyarray(surfacePairs, dtype=np.int64)

    def writeInput(self) -> str:

        out = '*SURFACE,NAME={:s}, TYPE=NODE\n'.format(self.name)

        for i in range(self._elSurfacePairs.shape[0]):
            out += '{:d},S{:d}\n'.format(self._elSurfacePairs[i,0], self._elSurfacePairs[i,1])

        #out += np.array2string(self.els, precision=2, separator=', ', threshold=9999999999)[1:-1]
        return out

class SurfaceSet(MeshSet):
    """
    A surface-set set is basic entity for storing element face lists, typically for setting directional fluxes onto
    surface elements based on the element ordering. The set remains constant without any dynamic referencing
    to any underlying geometric entities.
    """
    def __init__(self, name, surfacePairs):

        super().__init__(name)
        self._elSurfacePairs = np.asanyarray(surfacePairs, dtype=np.int64)

    @property
    def surfacePairs(self) -> np.array:
        """
        Elements with the associated face orientations are specified as Nx2 numpy array, with the first column being
        the element Id, and the second column the chosen face orientation
        """
        return self._elSurfacePairs

    @surfacePairs.setter
    def surfacePairs(self, surfacePairs):
        self._elSurfacePairs = np.asanyarray(surfacePairs, dtype=np.int64)

    def writeInput(self) -> str:

        out = '*SURFACE,NAME={:s}\n'.format(self.name)

        for i in range(self._elSurfacePairs.shape[0]):
            out += '{:d},S{:d}\n'.format(self._elSurfacePairs[i,0], self._elSurfacePairs[i,1])

        #out += np.array2string(self.els, precision=2, separator=', ', threshold=9999999999)[1:-1]
        return out


class Connector:
    """
     A Connector is a rigid connector between a set of nodes and an (optional) reference node.
     """
    def __init__(self, name, nodes, refNode = None):
        self.name = name
        self._refNode = refNode
        self._nodeset = None

    @property
    def refNode(self):
        """
        Reference Node ID
        """
        return self._refNode

    @refNode.setter
    def refNode(self, node):
        self._refNode = node

    @property
    def nodeset(self):
        """
        Nodes contains the list of Node IDs
        """
        return self._nodeset

    @nodeset.setter
    def nodeset(self, nodes):

        if isinstance(nodes, list) or isinstance(nodes,np.ndarray):
            self._nodeset = NodeSet('Connecter_{:s}'.format(self.name), np.array(nodes))
        elif isinstance(nodes,NodeSet):
            self._nodeset = nodes
        else:
            raise Exception('Invalid type for nodes passed to Connector()')

    def writeInput(self) -> str:
        # A nodeset is automatically created from the name of the connector
        strOut = '*RIGIDBODY, NSET={:s}'.format(self.nodeset.name)

        # A reference node is optional
        if isinstance(self.refNode, int):
            strOut += ',REF NODE={:d}\n'.format(self.refNode)
        else:
            strOut += '\n'

        return strOut


class DOF:
    """
    Provides a reference to the typical degrees-of-freedom (DOF) used for setting boundary conditions and displaying
    the required output in Calculix.
    """

    UX = 1
    """ Translation in the X direction """

    UY = 2
    """ Translation in the Y direction """

    UZ = 3
    """ Translation in the Z direction """

    RX = 4
    """ Rotation about the X-axis"""

    RY = 5
    """ Rotation about the Y-axis"""

    RZ = 6
    """ Rotation about the Z-axis"""

    T = 11
    """ Temperature """

