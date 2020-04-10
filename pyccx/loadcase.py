import numpy as np
import abc
import os
from enum import Enum, auto


class LoadCaseType(Enum):
    STATIC = auto()
    THERMAL = auto()
    UNCOUPLEDTHERMOMECHANICAL = auto()
    BUCKLE = auto()
    MODAL = auto()
    DYNAMIC = auto()


class LoadCase:
    def __init__(self, name):

        self._input = ''
        self.name = name

        self.isSteadyState = False
        self.loadCaseType = False
        self.initialTimeStep = 0.1
        self.defaultTimeStep = 0.1
        self.totalTime = 1.0

        self.resultSet = []
        self.boundaryConditions = []

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

    def setLoadCaseType(self, lType) -> None:
        """
        Set the loadcase type

        :param lType: Set the loadcase type using the enum LoadCaseType
        """
        if lType == LoadCaseType.STATIC:
            self.loadCaseType = LoadCaseType.STATIC
        elif lType == LoadCaseType.THERMAL:
            self.loadCaseType = LoadCaseType.THERMAL
        elif lType == LoadCaseType.UNCOUPLEDTHERMOMECHANICAL:
            self.loadCaseType = LoadCaseType.UNCOUPLEDTHERMOMECHANICAL
        else:
            raise ValueError('Load case type is not supported')

    def writeBoundaryCondition(self) -> str:
        """
        Generates the outString containing all the attached boundary conditions. Calculix cannot share existing boundary
        conditions and therefore has to be explicitly referenced per loadcase

        :return: outStr
        """
        bcondStr = ''

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
                        bcondStr += '{:s},{:d},, {:e}\n'.format(nodeset, bcond['dof'][i],
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

        if self.loadCaseType == LoadCaseType.STATIC:
            outStr += '*STATIC'
        elif self.loadCaseType == LoadCaseType.THERMAL:
            outStr += '*HEAT TRANSFER'
        elif self.loadCaseType == LoadCaseType.UNCOUPLEDTHERMOMECHANICAL:
            outStr += '*UNCOUPLED TEMPERATURE-DISPLACEMENT'

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
