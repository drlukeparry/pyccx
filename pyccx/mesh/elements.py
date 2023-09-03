import numpy as np

from abc import ABC, abstractmethod
from enum import Enum, IntEnum
from typing import List, Optional, Tuple

from .utils import classproperty

class ElementFamilies(IntEnum):
    """ Element Family Types"""
    Pnt     = 1
    Line    = 2
    Tri     = 3
    Quad    = 4
    Tet     = 5
    Pyramid = 6
    Prism   = 7
    Hex     = 8

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_

class ElementTypes(IntEnum):
    """ Element Family Types"""
    Node     = 1
    Line     = 2
    Planar   = 3
    Shell    = 4
    Axisymmetric = 5
    Volume   = 6

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


class BaseElementType(ABC):


    Type = None

    Data = {
        'NODE':  {'id': 15, 'name': 'Node', 'nodes': 1, 'order': 0, 'family': ElementFamilies.Pnt,
                  'faces': None, 'elementType': ElementTypes.Node},
        'BEAM2': {'id': 1, 'name': 'B31', 'nodes': 2, 'order': 1, 'family': ElementFamilies.Line,
                  'faces': [[1,2]], 'elementType': ElementTypes.Line},
        'BEAM3': {'id': 1, 'name': 'B32', 'nodes': 3, 'order': 2, 'family': ElementFamilies.Line,
                  'faces': [[1,2], [2,3]], 'elementType': ElementTypes.Line},
        'TRI3':  {'id': 2, 'name': 'CPS3', 'nodes': 3, 'order': 1, 'family': ElementFamilies.Tri,
                  'faces': [[1,2],[2,3],[3,1]], 'elementType': ElementTypes.Planar},
        'TRI6':  {'id': 9, 'name': 'CPS6', 'nodes': 3, 'order': 2, 'family': ElementFamilies.Tri,
                  'faces': [[1,2],[2,3],[3,1]], 'elementType': ElementTypes.Planar},
        'QUAD4': {'id': 3, 'name': 'CPS4', 'nodes': 4, 'order': 1, 'family': ElementFamilies.Quad,
                  'faces': [[1, 2], [2, 3], [3, 4], [4,1]], 'elementType': ElementTypes.Planar},
        'QUAD8': {'id': 16, 'name': 'CPS8', 'nodes': 8, 'order': 2, 'family': ElementFamilies.Quad,
                  'faces': [[1, 2], [2, 3], [3, 4], [4,1]],'elementType': ElementTypes.Planar},
        'SHELL3': {'id': 2, 'name': 'S3', 'nodes': 3, 'order': 1, 'family': ElementFamilies.Tri,
                   'faces': [[1,2],[2,3],[3,1]], 'elementType': ElementTypes.Shell},
        'SHELL4': {'id': 3, 'name': 'S4', 'nodes': 4, 'order': 1, 'family': ElementFamilies.Quad,
                   'faces': [[1, 2], [2, 3], [3, 4], [4,1]], 'elementType': ElementTypes.Shell},
        'SHELL8': {'id': 16, 'name': 'S8', 'nodes': 8, 'order': 2, 'family': ElementFamilies.Quad,
                   'faces': [[1, 8, 4, 7, 3, 6, 2, 5]], 'elementType': ElementTypes.Shell},
        'AX3':   {'id': 2, 'name': 'CAX3', 'nodes': 3, 'order': 1, 'family': ElementFamilies.Tri,
                  'faces': [[1, 2], [2, 3], [3, 1]], 'elementType': ElementTypes.Axisymmetric },
        'AX4':   {'id': 3, 'name': 'CAX4', 'nodes': 4, 'order': 1, 'family': ElementFamilies.Quad,
                  'faces': [[1, 2], [2, 3], [3, 4], [4,1]],
                  'elementType': ElementTypes.Axisymmetric
                  },
        'TET4':  {'id': 4, 'name': 'C3D4', 'nodes': 4, 'order': 1, 'family': ElementFamilies.Tet,
                  'faces': [[1, 2, 3], [1, 4, 2], [2, 4, 3], [3, 4, 1]],
                  'elementType': ElementTypes.Volume},
        'TET10': {'id': 11, 'name': 'C3D10', 'nodes': 10, 'order': 2, 'family': ElementFamilies.Tet,
                  'faces': [[1, 2, 3], [1, 4, 2], [2, 4, 3], [3, 4, 1]],
                  'elementType': ElementTypes.Volume
                  },
        'HEX8':  {'id': 5, 'name': 'C3D8', 'nodes': 8, 'order': 1, 'family': ElementFamilies.Hex,
                  'faces': [[1, 2, 3, 4], [5, 8, 7, 6], [1, 5, 6, 2], [2, 6, 7, 3], [3, 7, 8, 4], [4, 8, 5, 1]],
                  'elementType': ElementTypes.Volume
                  },
        'HEX8R': {'id': 5, 'name': 'C3D8R', 'nodes': 8, 'order': 1, 'family': ElementFamilies.Hex,
                   'faces': [[1, 2, 3, 4], [5, 8, 7, 6], [1, 5, 6, 2], [2, 6, 7, 3], [3, 7, 8, 4], [4, 8, 5, 1]],
                   'elementType': ElementTypes.Volume
                  },
        'HEX20': {'id': 17, 'name': 'C3D20', 'nodes': 20, 'order': 2, 'family': ElementFamilies.Hex,
                  'faces': [[1, 2, 3, 4], [5, 8, 7, 6], [1, 5, 6, 2], [2, 6, 7, 3], [3, 7, 8, 4], [4, 8, 5, 1]],
                  'elementType': ElementTypes.Volume
                  },
        'WEDGE6': {'id': 6, 'name': 'C3D6', 'nodes': 6, 'order': 1, 'family': ElementFamilies.Prism,
                   'faces': [[1,2,3], [4,5,6], [1,2,5,4], [2,3,6,5], [3,1,4,6]],
                   'elementType': ElementTypes.Volume
                   },
    }

    def __init__(self):
        pass

    @classproperty
    def name(cls):
        return cls.Data[cls.Type]['name']

    @classproperty
    def id(cls):
        return cls.Data[cls.Type]['id']

    @classproperty
    def nodes(cls) -> int:
        return cls.Data[cls.Type]['nodes']

    @classproperty
    def order(cls) -> int:
        return cls.Data[cls.Type]['order']

    @classproperty
    def faces(cls) -> List[List[int]]:
        return cls.Data[cls.Type]['faces']

    @classproperty
    def elementType(cls) -> List[List[int]]:
        return cls.Data[cls.Type]['elementType']

    @classproperty
    def faceMask(cls) -> List[List[int]]:

        nodeNum = cls.Data[cls.Type]['nodes']
        faceIds = cls.Data[cls.Type]['faces']
        mask = np.zeros([len(faceIds), nodeNum])

        for i, faceId in enumerate(faceIds):
            mask[i, np.array(faceId)-1] = 1

        return mask

    @classproperty
    def family(cls) -> ElementFamilies:
        return cls.Data[cls.Type]['family']

    @classmethod
    def elementType(cls):
        return cls.Data[cls.Type]

class NODE(BaseElementType):
    """ A single node element"""
    Type = 'NODE'

class BEAM2(BaseElementType):
    """ A Linear Beam Element  """
    Type = 'BEAM2'

class BEAM3(BaseElementType):
    """ A Quadratic Beam Element """
    Type = 'BEAM3'

class TET4(BaseElementType):
    """ 1st Order Linear Tet Element (C3D4) """
    Type = 'TET4'

class TET10(BaseElementType):
    """ 2nd order Quadratic Tet Element (C3D10) consisting of 10 nodes """
    Type = 'TET10'

class HEX8(BaseElementType):
    """ 1st order Linear Hexahedral Element (C3D8) """
    Type = 'HEX8'

class HEX8R(BaseElementType):
    """
    Linear Hex Element (C3D8I) with reformulation to reduce the effects of shear and
    volumetric locking and hourglass effects under some extreme situations
    """
    Type = 'HEX8R'

class HEX20(BaseElementType):
    """
    Quadratic Hexahedral Element (C3D20) consisting of 20 Nodes
    """
    Type = 'HEX20'

class WEDGE6(BaseElementType):
    """ 1st order Wedge or Prism Element (C3D6) """
    Type = 'WEDGE6'

class TRI3(BaseElementType):
    """ 1st order Tri Planar Stress Element (CPS4) """
    Type = 'TRI3'

class QUAD4(BaseElementType):
    """ 1st order Quad Planar Stress Element (CPS4) """
    Type = 'QUAD4'

class QUAD8(BaseElementType):
    """ 2nd order Quad Planar Stress Element (CPS4) """
    Type = 'QUAD8'

class SHELL3(BaseElementType):
    """ 1st order Tri Shell Element (S3) """
    Type = 'SHELL3'

class SHELL4(BaseElementType):
    """ 1st order Quad Shell Element (S4) """
    Type = 'SHELL4'

class SHELL8(BaseElementType):
    """ 2nd order Quad Shell Element (S8) """
    Type = 'SHELL8'

class AX3(BaseElementType):
    """ 1st order Axisymmetric Tri Element (CAX4) """
    Type = 'AX3'

class AX4(BaseElementType):
    """ 1st order Axisymmetric Quad Element (CAX4) """
    Type = 'AX4'

def elementTypes():
    """
    Returns the list of available element types available
    :return:
    """
    availableElementTypes = [NODE, BEAM2, BEAM3, TET10, HEX8, HEX20, HEX8R, SHELL3, SHELL4, SHELL8]

    return availableElementTypes

def getElementById(elTypeId: int):
    """
    Factory method for initialising an element class type

    :param elTypeId:
    :return:
    """

    for eType in elementTypes():
        if eType.id == elTypeId:
            return eType

    return None