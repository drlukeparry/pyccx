import numpy as np
import abc
from enum import Enum


class LoadCaseType(Enum):
    Static = 1
    Thermal = 2
    UnCoupledThermoMechanical = 4


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
        if lType == LoadCaseType.Static:
            self.loadCaseType = LoadCaseType.Static
        elif lType == LoadCaseType.Thermal:
            self.loadCaseType = LoadCaseType.Thermal
        elif lType == LoadCaseType.UnCoupledThermoMechanical:
            self.loadCaseType = LoadCaseType.UnCoupledThermoMechanical
        else:
            raise ValueError('Load case type is not supported')

    def writeBoundaryCondition(self) -> str:
        """
        Generates the string containing all the attached boundary conditions. Calculix cannot share existing boundary
        conditions and therefore has to be explicitly referenced per loadcase

        :return: str
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
                bcondStr += '{:s},GRAV,{:d}, {:.3f},{:.3f},{:.3f}\n'.format(bcond['el'], i, bcond['mag'], bcond['dir'][0],
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

    def writeInput(self):

        str = ''
        str += '{:*^64}\n'.format(' LOAD CASE ({:s}) '.format(self.name))
        str += '*STEP\n'
        # Write the thermal analysis loadstep

        if self.loadCaseType == LoadCaseType.Static:
            str += '*STATIC'
        elif self.loadCaseType == LoadCaseType.Thermal:
            str += '*HEAT TRANSFER'  # ',STEADY STATE
        elif self.loadCaseType == LoadCaseType.UnCoupledThermoMechanical:
            str += '*UNCOUPLED TEMPERATURE-DISPLACEMENT'  # ',STEADY STATE

        if self.isSteadyState:
            str += ', STEADY STATE'

        str += '\n{:.3f}, {:.3f}\n'.format(self.initialTimeStep, self.totalTime)

        str += self.writeBoundaryCondition()

        str += os.linesep
        for postResult in self.resultSet:
            str += postResult.writeInput()

        str += '*END STEP\n\n'

        return str
