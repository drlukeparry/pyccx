import numpy as np
import abc
import os

from enum import Enum, IntEnum, auto
from typing import List, Tuple, Type

from ..bc import BoundaryCondition, BoundaryConditionType
from ..core import ModelObject
from ..results import Result


class LoadCaseType(IntEnum):
    """
    Enum Class specifies the Load Case Type
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
    A unique Load case defines a set of simulation analysis conditions and a set of boundary conditions to apply to the domain.
    The default and initial timestep provide an estimate for the solver should be specified  along with the total duration
    of the load case using :meth:`setTimeStep`. The analysis type for the loadcase should be
    specified using :meth:`setLoadCaseType`. Depending on the analysis type the steady-state solution
    may instead be calculated.
    """
    def __init__(self, name: str, loadCaseType: LoadCaseType = None, resultSets = None):

        self._input = ''
        self._loadCaseType = None
        self._isSteadyState = False
        self._isNonlinear = False
        self._automaticIncrements = True
        self._initialTimestep = 0.1
        self._defaultTimeStep = 0.1
        self._minTimestep = 1e-6
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

        super().__init__(name)

    @property
    def loadCaseType(self) -> LoadCaseType:
        return self._loadCaseType

    @property
    def boundaryConditions(self) -> List[BoundaryCondition]:
        """
        The list of boundary conditions to be applied in the LoadCase
        """
        return self._boundaryConditions

    @boundaryConditions.setter
    def boundaryConditions(self, bConds: List[BoundaryCondition]):
        self._boundaryConditions = bConds

    @property
    def resultSet(self) -> List[Result]:
        """
        The result outputs (:class:`~pyccx.results.ElementResult`, :class:`~pyccx.results.NodeResult`) to generate
        the set of results from this loadcase.
        """
        return self._resultSet

    @resultSet.setter
    def resultSet(self, rSets: Type[Result]):
        if not any(isinstance(x, Result) for x in rSets):
            raise ValueError('Loadcase ResultSets must be derived from a Result class')
        else:
            self._resultSet = rSets

    @property
    def maxTimestep(self) -> float:
        return self._maxTimestep

    @maxTimestep.setter
    def maxTimestep(self, timeInc: float):
        self._maxTimestep = timeInc

    @property
    def minTimestep(self) -> float:
        return self._minTimestep

    @minTimestep.setter
    def minTimestep(self, timeInc: float):
        self._minTimestep = timeInc

    @property
    def totalTime(self) -> float:
        return self._totalTime

    @totalTime.setter
    def totalTime(self, time: float):
        self._totalTime = time

    @property
    def defaultTimestep(self) -> float:
        return self._defaultTimestep

    @defaultTimestep.setter
    def defaultTimestep(self, timestep: float):
        self._defaultTimestep = timestep

    @property
    def initialTimestep(self):
        return self._initialTimestep

    @initialTimestep.setter
    def initialTimestep(self, timeStep: float):
        self._initialTimestep = timeStep

    @property
    def steadyState(self) -> bool:
        """
        Returns True if the loadcase is a steady-state analysis
        """
        return self._isSteadyState

    @steadyState.setter
    def steadyState(self,state:bool) -> None:
        self._isSteadyState = state

    @property
    def automaticIncrements(self) -> bool:
        return self._automaticIncrements

    @automaticIncrements.setter
    def automaticIncrements(self, state : bool):
        self._automaticIncrements = state

    @property
    def nonlinear(self):
        return self._isNonlinear

    @nonlinear.setter
    def nonlinear(self, state):
        self._isNonlinear = state

    def setTimeStep(self, defaultTimestep: float = 1.0, initialTimestep: float = None, totalTime: float = None) -> None:
        """
        Set the timestepping values for the loadcase

        :param defaultTimestep: float: Default timestep to use throughout the loadcase
        :param initialIimestep:  float: The initial timestep to use for the increment
        :param totalTime: float: The total time for the loadcase

        """
        self._defaultTimestep = defaultTimestep

        if initialTimestep:
            self._initialTimestep = initialTimestep

        if totalTime:
            self._totalTime = totalTime

    def setLoadCaseType(self, loadCaseType: LoadCaseType) -> None:
        """
        Set the loadcase type based on the analysis types available in :class:`~pyccx.loadcase.LoadCaseType`.

        :param loadCaseType: Set the load case type using the enum :class:`~pyccx.loadcase.LoadCaseType`
        """

        if isinstance(loadCaseType, LoadCaseType):
            self._loadCaseType = loadCaseType
        else:
            raise ValueError('Load case type is not supported')

    def writeBoundaryCondition(self) -> str:
        """
        Generates the string for Boundary Conditions in self.boundaryConditions containing all the attached boundary
        conditions. Calculix cannot share existing boundary conditions and therefore has to be explicitly
        created per load case.

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

        # Write the thermal analysis loadstep

        if self._loadCaseType == LoadCaseType.STATIC:
            outStr += '*STATIC'
        elif self._loadCaseType == LoadCaseType.THERMAL:
            outStr += '*HEAT TRANSFER'
        elif self._loadCaseType == LoadCaseType.UNCOUPLEDTHERMOMECHANICAL:
            outStr += '*UNCOUPLED TEMPERATURE-DISPLACEMENT'
        else:
            raise ValueError('The type ({:s}) for Loadcase ({:s})  is not currently supported in PyCCX'.format(self._loadCaseType,
                                                                                                               self.name))
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

