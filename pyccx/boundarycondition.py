import abc
from enum import Enum, auto
import numpy as np

from .core import ElementSet, NodeSet, SurfaceSet


class BoundaryConditionType(Enum):
    """
    Boundary condition type specifies which type of analyses the boundary condition may be applied to.
    """
    ANY = auto()
    STRUCTURAL = auto()
    THERMAL = auto()
    FLUID = auto()


class BoundaryCondition(abc.ABC):
    """
    Base class for all boundary conditions
    """

    def __init__(self, model, target):

        self.init = True
        self.model  = model
        self.target = target

    def getBoundaryElements(self):

        if isinstance(self.target, ElementSet):
            return self.target.els

        return None

    def getBoundaryFaces(self):

        if isinstance(self.target, SurfaceSet):
            return self.target.nodes

        return None

    def getBoundaryNodes(self):

        if isinstance(self.target,NodeSet):
            return self.target.nodes

        return None

    @abc.abstractmethod
    def type(self) -> int:
        """
        Returns the BC type so that they are only applied to suitable load cases
        """
        raise NotImplemented()

    @abc.abstractmethod
    def writeInput(self) -> str:
        raise NotImplemented()


class Film(BoundaryCondition):
    """
    The film or convective heat transfer boundary condition applies the Newton's law of cooling
    - :math:`q = h_{c}\\left(T-T_{amb}\\right)` to specified faces of
    boundaries elements (correctly ordered according to Calculix's requirements). This BC may be used in thermal and
    coupled thermo-mechanical analyses.
    """

    def __init__(self, model, faces):

        self.h = 0.0
        self.T_amb = 0.0
        self.faces = faces

        super().__init__(model, faces)

    def type(self) -> int:
        return BoundaryConditionType.THERMAL

    @property
    def heatTransferCoefficient(self) -> float:
        """
        The heat transfer coefficient :math:`h_{c}` used for the Film Boundary Condition
        """
        return self.h

    @heatTransferCoefficient.setter
    def heatTransferCoefficient(self, h: float) -> None:
        self.h = h

    @property
    def ambientTemperature(self) -> float:
        """
        The ambient temperature :math:`T_{amb}`. used for the Film Boundary Condition
        """
        return self.T_amb

    @heatTransferCoefficient.setter
    def ambientTemerature(self, Tamb:float) -> None:
        self.T_amb = Tamb

    def writeInput(self) -> str:
        bCondStr = '*FILM\n'

        bfaces = self.getBoundaryFaces()

        for i in len(bfaces):
            bCondStr += '{:d},F{:d},{:e},{:e}\n'.format(bfaces[i, 0], bfaces[i, 1], self.T_amb, self.h)

        return bCondStr


class HeatFlux(BoundaryCondition):
    """
    The flux boundary condition applies a uniform external heat flux :math:`q` to faces of surface
    boundaries elements (correctly ordered according to Calculix's requirements). This BC may be used in thermal and
    coupled thermo-mechanical analyses.
    """

    def __init__(self, model, faces):

        self.flux = 0.0
        self.faces = faces

        super().__init__(model, faces)


    def type(self) -> int:
        return BoundaryConditionType.THERMAL

    @property
    def flux(self) -> float:
        """
        The flux value :math:`q` used for the Heat Flux Boundary Condition
        """
        return self.flux

    @flux.setter
    def flux(self, fluxVal: float) -> None:
        self.flux = fluxVal

    def writeInput(self) -> str:

        bCondStr = '*DFLUX\n'
        bfaces = self.getBoundaryFaces()

        for i in range(len(bfaces)):
            bCondStr += '{:d},S{:d},{:e}\n'.format(bfaces[i, 0], bfaces[i, 1], self.flux)

        return bCondStr


class Radiation(BoundaryCondition):
    """
    The radiation boundary condition applies Black-body radiation using the Stefan-Boltzmann Law,
    :math:`q_{rad} = \\epsilon \\sigma_b\\left(T-T_{amb}\\right)^4`, which is imposed on the faces of
    boundaries elements (correctly ordered according to Calculix's requirements). Ensure that the Stefan-Boltzmann constant :math:
    `\\sigma_b`, has consistent units, which is set in the :attribute:`~pyccx.core.Simulation.SIGMAB`. This BC may be used in thermal and
    coupled thermo-mechanical analyses.
    """

    def __init__(self, model, faces):

        self.T_amb = 0.0
        self.epsilon = 1.0
        self.faces = faces

        super().__init__(model, faces)

    def type(self) -> int:
        return BoundaryConditionType.THERMAL

    @property
    def emmisivity(self) -> float:
        """
        The emmisivity value :math:`\\epsilon` used for the Radiation Boundary Condition
        """
        return self.emmisivity

    @property
    def ambientTemperature(self) -> float:
        """
        The ambient temperature :math:`T_{amb}`. used for the Radiation Boundary Condition
        """
        return self.T_amb

    @ambientTemperature.setter
    def ambientTemerature(self, Tamb: float) -> None:
        self.T_amb = Tamb


    def writeInput(self) -> str:

        bCondStr = '*RADIATE\n'
        bfaces = self.getBoundaryFaces()

        for i in range(len(bfaces)):
            bCondStr += '{:d},F{:d},{:e},{:e}\n'.format(bfaces[i, 0], bfaces[i, 1], self.T_amb, self.epsilon)

        return bCondStr


class Fixed(BoundaryCondition):
    """
    The fixed boundary condition removes or sets the DOF (e.g. displacement components, temperature) specifically on
    a Node Set. This BC may be used in thermal and coupled thermo-mechanical analyses provided the DOF is applicable to
    the analysis type.
    """

    def __init__(self, model, target):

        self.dof = []
        self.values = []

        super().__init__(model, target)

    def type(self) -> int:
        return BoundaryConditionType.ANY

    def writeInput(self) -> str:

        raise NotImplemented()

        bCondStr = '*BOUNDARY\n'
        bcond = {}

        nodeset = self.getBoundaryNodes()
        # 1-3 U, 4-6, rotational DOF, 11 = Temp
        for i in range(len(bcond['dof'])):
            if 'value' in bcond.keys():
                # inhomogenous boundary conditions
                bCondStr += '{:s},{:d},, {:e}\n'.format(nodeset, bcond['dof'][i],
                                                        bcond['value'][i])
            else:
                # Fixed boundary condition
                bCondStr += '{:s},{:d}\n'.format(nodeset, bcond['dof'][i])

        return bCondStr


class Acceleration(BoundaryCondition):
    """
    The Acceleration Boundary Condition applies an acceleration term across a Volume (i.e. Element Set) during a structural
    analysis. This is provided as magnitude, direction of the acceleration on the body.
    """

    def __init__(self, model, target):

        self.mag = 1.0
        self.dir = np.array([0.0,0.0,1.0])
        super().__init__(model, target)

    def type(self) -> int:
        return BoundaryConditionType.STRUCTURAL

    def setVector(self, v) -> None:
        """
        The acceleration of the body set by an Acceleration Vector

        :param v: The vector of the acceleration
        """
        from numpy import linalg
        mag = linalg.norm(v)
        self.dir = v / linalg.norm(v)
        self.magnitude = mag

    @property
    def magnitude(self) -> float:
        """
        The acceleration magnitude applied onto the body
        """
        return self.mag

    @magnitude.setter
    def magnitude(self, magVal: float) -> None:
        from numpy import linalg
        self.mag = magVal

    @property
    def direction(self) -> float:
        """
        The acceleration direction (noramlised vector)
        """
        return self.dir

    @direction.setter
    def direction(self, v: float) -> None:
        from numpy import linalg
        self.dir = v / linalg.norm(v)


    def writeInput(self) -> str:
        bCondStr = '*DLOAD\n'
        bCondStr += '{:s},GRAV,{:d}, {:.3f},{:.3f},{:.3f}\n'.format(self.getBoundaryElements(), self.mag,*self.dir)
        return bCondStr


class Pressure(BoundaryCondition):
    """
    The Pressure Boundary Condition applies a uniform pressure to faces across an element boundary.
    """

    def __init__(self, model, faces):

        self.mag = 0.0
        self.faces = faces
        super().__init__(model, faces)

    def type(self) -> int:
        return BoundaryConditionType.STRUCTURAL

    @property
    def magnitude(self) -> float:
        """
        The pressure magnitude applied onto the surface
        """
        return self.mag

    @magnitude.setter
    def magnitude(self, magVal: float) -> None:
        self.mag = magVal

    def writeInput(self) -> str:

        bCondStr = '*DLOAD\n'
        bfaces = self.getBoundaryFaces()

        for i in range(len(bfaces)):
            bCondStr += '{:d},P{:d},{:e}\n'.format(bfaces[i, 0], bfaces[i, 1], self.mag)
            
        return bCondStr

class Force(BoundaryCondition):
    """
    The Force Boundary applies a uniform force directly to nodes. This BC may be used in thermal and
    coupled thermo-mechanical analyses provided the DOF is applicable to the analysis type.
    """

    def __init__(self, model, target):

        self.mag = 0.0
        self.dir = np.array([0.0,0.0,1.0])
        super().__init__(model, target)

    def type(self) -> int:
        return BoundaryConditionType.STRUCTURAL

    def setVector(self, v) -> None:
        """
        The applied force set by the vector

        :param v: The Force vector
        """
        from numpy import linalg
        mag = linalg.norm(v)
        self.dir = v / linalg.norm(v)
        self.magnitude = mag

    @property
    def magnitude(self) -> float:
        """
        The magnitude of the force applied
        """
        return self.mag

    @magnitude.setter
    def magnitude(self, magVal: float) -> None:
        self.mag = magVal

    @property
    def direction(self) -> float:
        """
        The normalised vector of the force direction
        """
        return self.dir

    @direction.setter
    def direction(self, v: float) -> None:
        from numpy import linalg
        self.dir = v / linalg.norm(v)

    def writeInput(self) -> str:

        bCondStr = '*CLOAD\n'
        nodeset = self.getBoundaryNodes()

        for i in range(3):
            compMag = self.mag * self.dir[i]
            bCondStr += '{:s},{:d}\n'.format(nodeset, i, compMag)

        return bCondStr
