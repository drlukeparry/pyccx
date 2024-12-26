import numpy as np
import os

from enum import IntEnum, auto
from typing import List, Tuple, Type, Optional

from ..bc import BoundaryCondition, BoundaryConditionType
from ..core import ModelObject
from ..results import Result


class LoadCaseType(IntEnum):
    """
    Enum Class specifies the Load Case type
    """

    STATIC = auto()
    """Linear Static structural analysis"""

    THERMAL = auto()
    """Thermal analysis for performing heat transfer studies"""

    UNCOUPLEDTHERMOMECHANICAL = auto()
    """Coupled thermo-mechanical analysis"""

    BUCKLE = auto()
    """Buckling analysis of a structure"""

    MODAL = auto()
    """Modal analysis of a structure"""

    DYNAMIC = auto()
    """Dynamic analysis of a structure"""


class LoadCase(ModelObject):
    """
    A unique Load case defines a set of simulation analysis conditions and a set of boundary conditions to apply to
    the domain. The default and initial timestep provide an estimate for the solver should be specified  along with
    the total duration of the load case using :meth:`setTimeStep`. The analysis type for the loadcase should be
    specified using :meth:`setLoadCaseType`. Depending on the analysis type the steady-state solution may instead be
    calculated.

    If the option :attr:`automaticIncrements` is set to False, the solver will use the initial timestep and the total
    time steps will be defined by the user when using the :attr:`nonlinear` option, which provides time-dependent
    behavior for the analysis.

    """
    def __init__(self, name: str,
                 loadCaseType: Optional[LoadCaseType] = None,
                 resultSets: Optional[List[Result]] = None):

        super().__init__(name)

        # Internal output
        self._input = ''

        """ Analysis Types for the Load Case"""
        self._loadCaseType = None
        self._isSteadyState = False
        self._isNonlinear = False

        """ Time-stepping parameters """
        self._automaticIncrements = True
        self._initialTimestep = 0.1
        self._defaultTimestep = 0.1
        self._minTimestep  = 1e-6
        self._maxTimestep = 1.0
        self._totalTime = 1.0
        self._resultSet = []
        self._boundaryConditions = []

        if loadCaseType:
            if loadCaseType in LoadCaseType:
                self._loadCaseType = loadCaseType
            else:
                raise ValueError('Loadcase type must valid')

        if resultSets:
            self.resultSet = resultSets

    @property
    def loadCaseType(self) -> LoadCaseType:
        return self._loadCaseType

    @property
    def boundaryConditions(self) -> List[BoundaryCondition]:
        """
        The list of boundary conditions to be applied during the load case
        """
        return self._boundaryConditions

    @boundaryConditions.setter
    def boundaryConditions(self, bConds: List[BoundaryCondition]):
        self._boundaryConditions = bConds

    @property
    def resultSet(self) -> List[Result]:
        """
        The result outputs (:class:`~pyccx.results.ElementResult`, :class:`~pyccx.results.NodeResult`) to generate
        the set of results from this load case.
        """
        return self._resultSet

    @resultSet.setter
    def resultSet(self, rSets: List[Result]):
        if not any(isinstance(x, Result) for x in rSets):
            raise ValueError('Loadcase ResultSets must be derived from a Result class')
        else:
            self._resultSet = rSets

    @property
    def maxTimestep(self) -> float:
        """
        The maximum timestep increment for the load case if the solver is using an adaptive time-stepping scheme
        which is used during a non-linear or incremental loading.
        """
        return self._maxTimestep

    @maxTimestep.setter
    def maxTimestep(self, timeInc: float):
        self._maxTimestep = timeInc

    @property
    def minTimestep(self) -> float:
        """
        The minimum timestep increment for the load case if the solver is using an adaptive time-stepping scheme
        """
        return self._minTimestep

    @minTimestep.setter
    def minTimestep(self, timeInc: float):
        self._minTimestep = timeInc

    @property
    def totalTime(self) -> float:
        """
        The total time duration for the load case
        """
        return self._totalTime

    @totalTime.setter
    def totalTime(self, time: float):
        self._totalTime = time

    @property
    def defaultTimestep(self) -> float:
        """
        The default timestep to use throughout the load case
        """
        return self._defaultTimestep

    @defaultTimestep.setter
    def defaultTimestep(self, timestep: float) -> None:
        self._defaultTimestep = timestep

    @property
    def initialTimestep(self) -> float:
        """
        The initial timestep to use for the increment during the load case if the solver is using an
        adaptive time-stepping scheme
        """
        return self._initialTimestep

    @initialTimestep.setter
    def initialTimestep(self, timeStep: float):
        self._initialTimestep = timeStep

    @property
    def steadyState(self) -> bool:
        """
        `True` if the loadcase is a steady-state analysis
        """
        return self._isSteadyState

    @steadyState.setter
    def steadyState(self, state: bool) -> None:
        self._isSteadyState = state

    @property
    def automaticIncrements(self) -> bool:
        """
        `True` if the solver is using adaptive time-stepping increments
        """
        return self._automaticIncrements

    @automaticIncrements.setter
    def automaticIncrements(self, state: bool) -> None:
        self._automaticIncrements = state

    @property
    def nonlinear(self) -> bool:
        """
        `True` if the load case is a non-linear analysis
        """
        return self._isNonlinear

    @nonlinear.setter
    def nonlinear(self, state) -> None:
        self._isNonlinear = state

    def setTimeStep(self,
                    defaultTimestep: float = 1.0,
                    initialTimestep: Optional[float] = None,
                    totalTime: Optional[float] = None) -> None:
        """
        Set the time stepping values for the loadcase

        :param defaultTimestep: float: Default timestep to use throughout the load case
        :param initialTimestep: float: The initial timestep to use for the increment
        :param totalTime: float: The total time for the load case

        """
        self._defaultTimestep = defaultTimestep

        if initialTimestep:
            self._initialTimestep = initialTimestep

        if totalTime:
            self._totalTime = totalTime

    def setLoadCaseType(self, loadCaseType: LoadCaseType) -> None:
        """
        Set the load case type based on the analysis types available in :class:`~pyccx.loadcase.LoadCaseType`.

        :param loadCaseType: Set the load case type using the enum :class:`~pyccx.loadcase.LoadCaseType`
        """

        if isinstance(loadCaseType, LoadCaseType):
            self._loadCaseType = loadCaseType
        else:
            raise ValueError('The load case type is not supported')

    def writeBoundaryCondition(self) -> str:
        """
        Generates the string for Boundary Conditions in self.boundaryConditions containing all the attached boundary
        conditions. Calculix cannot share existing boundary conditions and therefore has to be explicitly
        initialised and created per individual load case.

        :return: outStr
        """
        bCondStr = ''

        for bcond in self.boundaryConditions:
            bCondStr += bcond.writeInput()
            bCondStr += '\n'

        return bCondStr

    def writeInput(self) -> str:

        outStr = '\n'
        outStr += '{:*^80}\n'.format(' LOAD CASE ({:s}) '.format(self.name))
        outStr += '*STEP'

        if self._isNonlinear:
            outStr += ', NLGEOM=YES'

        outStr += '\n'

        # Write the  analysis loadstep case
        if self._loadCaseType == LoadCaseType.STATIC:
            outStr += '*STATIC'
        elif self._loadCaseType == LoadCaseType.THERMAL:
            outStr += '*HEAT TRANSFER'
        elif self._loadCaseType == LoadCaseType.UNCOUPLEDTHERMOMECHANICAL:
            outStr += '*UNCOUPLED TEMPERATURE-DISPLACEMENT'
        else:
            raise ValueError('The type ({:s}) for Loadcase ({:s}) is not currently supported in PyCCX'.format(self._loadCaseType, self.name))
        if self._isSteadyState:
            outStr += ', STEADY STATE'

        if not self._automaticIncrements:
            outStr += ', DIRECT'

        outStr += '\n'

        # Write the timestepping information
        outStr += '{:.7f}, {:.7f} ,{:.7f} , {:.7f}\n'.format(self._initialTimestep,
                                                             self._totalTime,
                                                             self._minTimestep, self._maxTimestep)

        outStr += '\n'
        # Write the individual boundary conditions associated with this loadcase
        outStr += self.writeBoundaryCondition()

        outStr += os.linesep
        for postResult in self.resultSet:
            outStr += postResult.writeInput()

        outStr += '*END STEP\n\n'

        return outStr
