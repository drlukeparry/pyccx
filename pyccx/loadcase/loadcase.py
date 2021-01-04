import numpy as np
import abc
import os

from enum import Enum, auto
from typing import List, Tuple, Type

from ..bc import BoundaryCondition, BoundaryConditionType
from ..results import Result


class LoadCaseType(Enum):
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


class LoadCase:
    """
    A unique Load case defines a set of simulation analysis conditions and a set of boundary conditions to apply to the domain.
    The default and initial timestep provide an estimate for the solver should be specified  along with the total duration
    of the load case using :meth:`setTimeStep`. The analysis type for the loadcase should be
    specified using :meth:`setLoadCaseType`. Depending on the analysis type the steady-state solution
    may instead be calculated.
    """
    def __init__(self, loadCaseName, loadCaseType: LoadCaseType = None, resultSets = None):

        self._input = ''
        self._loadcaseName = loadCaseName
        self._loadCaseType = None
        self.isSteadyState = False
        self.initialTimeStep = 0.1
        self.defaultTimeStep = 0.1
        self.totalTime = 1.0
        self._resultSet = []
        self._boundaryConditions = []

        if loadCaseType:
            if loadCaseType is LoadCaseType:
                self._loadCaseType = loadCaseType
            else:
                raise  ValueError('Loadcase type must valid')

        if resultSets:
            self.resultSet = resultSets

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
    def name(self) -> str:
        return self._loadcaseName

    @name.setter
    def name(self, loadCaseName):
        self._loadcaseName = loadCaseName

    @property
    def steadyState(self) -> bool:
        """
        Returns True if the loadcase is a steady-state analysis
        """
        return self.isSteadyState

    @steadyState.setter
    def steadyState(self,state:bool) -> None:
        self.isSteadyState = state


    def setTimeStep(self, defaultTimeStep: float = 1.0, initialIimeStep: float = None, totalTime: float = None) -> None:
        """
        Set the timestepping values for the loadcase

        :param defaultTimeStep: float: Default timestep to use throughout the loadcase
        :param initialIimeStep:  float: The initial timestep to use for the increment
        :param totalTime: float: The total time for the loadcase

        """
        self.defaultTimeStep = defaultTimeStep

        if initialIimeStep is not None:
            self.initialTimeStep = initialIimeStep

        if totalTime is not None:
            self.totalTime = totalTime

    def setLoadCaseType(self, loadCaseType: LoadCaseType) -> None:
        """
        Set the loadcase type based on the analysis types available in :class:`~pyccx.loadcase.LoadCaseType`.

        :param loadCaseType: Set the loadcase type using the enum :class:`~pyccx.loadcase.LoadCaseType`
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
        bcondStr = ''

        for bcond in self.boundaryConditions:
            bcondStr += bcond.writeInput()

        if False:
            for bcond in self.boundaryConditions:

                if bcond['type'] == 'film':

                    bcondStr += '*FILM\n'
                    bfaces = bcond['faces']
                    for i in len(bfaces):
                        bcondStr += '{:d},F{:d},{:e},{:e}\n'.format(bfaces[i, 0], bfaces[i, 1], bcond['tsink'], bcond['h'])

                elif bcond['type'] == 'bodyflux':

                    bcondStr += '*DFLUX\n'
                    bcondStr += '{:s},BF,{:e}\n'.format(bcond['el'], bcond['flux'])  # use element set

                elif bcond['type'] == 'faceflux':

                    bcondStr += '*DFLUX\n'
                    bfaces = bcond['faces']
                    for i in range(len(bfaces)):
                        bcondStr += '{:d},S{:d},{:e}\n'.format(bfaces[i, 0], bfaces[i, 1], bcond['flux'])

                elif bcond['type'] == 'radiation':

                    bcondStr += '*RADIATE\n'
                    bfaces = bcond['faces']
                    for i in len(bfaces):
                        bcondStr += '{:d},F{:d},{:e},{:e}\n'.format(bfaces[i, 0], bfaces[i, 1], bcond['tsink'],
                                                               bcond['emmisivity'])

                elif bcond['type'] == 'fixed':

                    bcondStr += '*BOUNDARY\n'
                    nodeset = bcond['nodes']
                    # 1-3 U, 4-6, rotational DOF, 11 = Temp

                    for i in range(len(bcond['dof'])):
                        if 'value' in bcond.keys():
                            bcondStr += '{:s},{:d},,{:e}\n'.format(nodeset, bcond['dof'][i],
                                                               bcond['value'][i])  # inhomogenous boundary conditions
                        else:
                            bcondStr += '{:s},{:d}\n'.format(nodeset, bcond['dof'][i])

                elif bcond['type'] == 'accel':

                    bcondStr += '*DLOAD\n'
                    bcondStr += '{:s},GRAV,{:.3f}, {:.3f},{:.3f},{:.3f}\n'.format(bcond['el'], bcond['mag'], bcond['dir'][0],
                                                                           bcond['dir'][1], bcond['dir'][2])

                elif bcond['type'] == 'force':

                    bcondStr += '*CLOAD\n'
                    nodeset = bcond['nodes']

                    for i in bcond['dof']:
                        bcondStr += '{:s},{:d}\n'.format(nodeset, i, bcond['mag'])

                elif bcond['type'] == 'pressure':

                    bcondStr += '*DLOAD\n'
                    bfaces = bcond['faces']
                    for i in range(len(bfaces)):
                        bcondStr += '{:d},P{:d},{:e}\n'.format(bfaces[i, 0], bfaces[i, 1], bcond['mag'])

        return bcondStr

    def writeInput(self) -> str:

        outStr  = '{:*^64}\n'.format(' LOAD CASE ({:s}) '.format(self.name))
        outStr += '*STEP\n'
        # Write the thermal analysis loadstep

        if self._loadCaseType == LoadCaseType.STATIC:
            outStr += '*STATIC'
        elif self._loadCaseType == LoadCaseType.THERMAL:
            outStr += '*HEAT TRANSFER'
        elif self._loadCaseType == LoadCaseType.UNCOUPLEDTHERMOMECHANICAL:
            outStr += '*UNCOUPLED TEMPERATURE-DISPLACEMENT'
        else:
            raise ValueError('Loadcase type ({:s} is not currently supported in PyCCX'.format(self._loadCaseType))

        if self.isSteadyState:
            outStr += ', STEADY STATE'

        # Write the timestepping information
        outStr += '\n{:.3f}, {:.3f}\n'.format(self.initialTimeStep, self.totalTime)

        # Write the individual boundary conditions associated with this loadcase
        outStr += self.writeBoundaryCondition()

        outStr += os.linesep
        for postResult in self.resultSet:
            outStr += postResult.writeInput()

        outStr += '*END STEP\n\n'

        return outStr
