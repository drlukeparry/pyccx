import re  # used to get info from frd file
import os
import sys
import subprocess  # used to check ccx version
from enum import Enum, auto
from typing import List, Tuple, Type
import logging

from ..bc import BoundaryCondition
from ..core import MeshSet, ElementSet, SurfaceSet, NodeSet, Connector
from ..loadcase import LoadCase
from ..material import Material
from ..mesh import Mesher
from ..results import ElementResult, NodalResult, ResultProcessor


class AnalysisError(Exception):
    """Exception raised for errors generated during the analysis

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message


class AnalysisType(Enum):
    """
    The analysis types available for use.
    """
    STRUCTURAL = auto()
    THERMAL = auto()
    FLUID = auto()


class Simulation:
    """
    Provides the base class for running a Calculix simulation
    """

    NUMTHREADS = 1
    """ Number of Threads used by the Calculix Solver """

    CALCULIX_PATH = ''
    """ The Calculix directory path used for Windows platforms"""

    VERBOSE_OUTPUT = True
    """ When enabled, the output during the analysis is redirected to the console"""

    def __init__(self, meshModel: Mesher):

        self._input = ''
        self._workingDirectory = ''
        self._analysisCompleted = False

        self._name = ''
        self.initialConditions = []  # 'dict of node set names,
        self._loadCases = []
        self._mpcSets = []
        self._connectors = []
        self._materials = []
        self._materialAssignments = []
        self.model = meshModel

        self.initialTimeStep = 0.1
        self.defaultTimeStep = 0.1
        self.totalTime = 1.0
        self.useSteadyStateAnalysis = True

        self.TZERO = -273.15
        self.SIGMAB = 5.669E-8
        self._numThreads = 1

        # Private sets are used for the storage of additional user defined sets
        self._surfaceSets = []
        self._nodeSets = []
        self._elementSets = []

        self.includes = []

    def init(self):

        self._input = ''

    @classmethod
    def setNumThreads(cls, numThreads: int):
        """
        Sets the number of simulation threads to use in Calculix

        :param numThreads:
        :return:
        """
        cls.NUMTHREADS = numThreads

    @classmethod
    def getNumThreads(cls) -> int:
        """
        Returns the number of threads used

        :return: int:
        """
        return cls.NUMTHREADS

    @classmethod
    def setCalculixPath(cls, calculixPath: str) -> None:
        """
        Sets the path for the Calculix executable. Necessary when using Windows where there is not a default
        installation proceedure for Calculix

        :param calculixPath: Directory containing the Calculix Executable
        """

        if os.path.isdir(calculixPath) :
            cls.CALCULIX_PATH = calculixPath

    @classmethod
    def setVerboseOuput(cls, state: bool) -> None:
        """
        Sets if the output from Calculix should be verbose i.e. printed to the console

        :param state:
        """

        cls.VERBOSE_OUTPUT = state

    def setWorkingDirectory(self, workDir) -> None:
        """
        Sets the working directory used during the analysis.

        :param workDir: An accessible working directory path

        """
        if os.path.isdir(workDir) and os.access(workDir, os.W_OK):
            self._workingDirectory = workDir
        else:
            raise ValueError('Working directory ({:s}) is not accessible or writable'.format(workDir))

    @property
    def name(self) -> str:
        return self._name

    def getBoundaryConditions(self) -> List[BoundaryCondition]:
        """
        Collects all :class:`~pyccx.boundarycondition.BoundaryCondition` which are attached to :class:`LoadCase` in
        the analysis

        :return:  All the boundary conditions in the analysis
        """
        bcs = []
        for loadcase in self._loadCases:
            bcs += loadcase.boundaryConditions

        return bcs

    @property
    def loadCases(self) -> List[LoadCase]:
        """
        List of :class:`~pyccx.loadcase.LoadCase` used in the analysis
        """
        return self._loadCases

    @loadCases.setter
    def loadCases(self, loadCases: List[LoadCase]):
        self._loadCases = loadCases

    @property
    def connectors(self) -> List[Connector]:
        """
        List of :class:`~pyccx.core.Connector` used in the analysis
        """
        return self._connectors

    @connectors.setter
    def connectors(self, connectors: List[Connector]):
        self._connectors = connectors

    @property
    def mpcSets(self):
        return self._mpcSets

    @mpcSets.setter
    def mpcSets(self, value):
        self._mpcSets = value

    @property
    def materials(self) -> List[Material]:
        """
        User defined :class:`~pyccx.material.Material` used in the analysis
        """
        return self._materials

    @materials.setter
    def materials(self, materials):
        self._materials = materials

    @property
    def materialAssignments(self):
        """
        Material Assignment applied to a set of elements
        """
        return self._materialAssignments

    @materialAssignments.setter
    def materialAssignments(self, matAssignments):
        self._materialAssignments = matAssignments

    def _collectSets(self, setType: Type[MeshSet] = None):
        """
        Private function returns a unique set of Element, Nodal, Surface sets which are used by the analysis during writing.
        This reduces the need to explicitly attach them to an analysis.
        """
        elementSets = {}
        nodeSets = {}
        surfaceSets = {}

        # Iterate through all user defined sets
        for elSet in self._elementSets:
            elementSets[elSet.name] = elSet

        for nodeSet in self._nodeSets:
            nodeSets[nodeSet.name] = nodeSet

        for surfSet in self._surfaceSets:
            surfaceSets[surfSet.name] = surfSet

        # Iterate through all loadcases and boundary conditions.and find unique values. This is greedy so will override
        # any with same name.
        for loadcase in self.loadCases:

            # Collect result sets node and element sets automatically
            for resultSet in loadcase.resultSet:
                if isinstance(resultSet, ElementResult):
                    elementSets[resultSet.elementSet.name] = resultSet.elementSet
                elif isinstance(resultSet, NodalResult):
                    nodeSets[resultSet.nodeSet.name] = resultSet.nodeSet

            for bc in loadcase.boundaryConditions:
                if isinstance(bc.target, ElementSet):
                    elementSets[bc.target.name] = bc.target

                if isinstance(bc.target, NodeSet):
                    nodeSets[bc.target.name] = bc.target

                if isinstance(bc.target, SurfaceSet):
                    surfaceSets[bc.target.name] = bc.target

        for con in self.connectors:
            nodeSets[con.nodeset.name] = con.nodeset

        if setType is ElementSet:
            return list(elementSets.values())
        elif setType is NodeSet:
            return list(nodeSets.values())
        elif setType is SurfaceSet:
            return list(surfaceSets.values())
        else:
            return list(elementSets.values()), list(nodeSets.values()), list(surfaceSets.values())

    @property
    def elementSets(self) -> List[ElementSet]:
        """
        User-defined :class:`~pyccx.core.ElementSet` manually added to the analysis
        """
        return self._elementSets

    @elementSets.setter
    def elementSets(self, val: List[ElementSet]):
        self._elementSets = val

    @property
    def nodeSets(self) -> List[NodeSet]:
        """
        User-defined :class:`~pyccx.core.NodeSet` manually added to the analysis
        """
        return self._nodeSets

    @nodeSets.setter
    def nodeSets(self, val: List[NodeSet]):
        nodeSets = val

    @property
    def surfaceSets(self) -> List[SurfaceSet]:
        """
        User-defined :class:`pyccx.core.SurfaceSet`  manually added to the analysis
        """
        return self._nodeSets

    @surfaceSets.setter
    def surfaceSets(self, val=List[SurfaceSet]):
        surfaceSets = val

    def getElementSets(self) -> List[ElementSet]:
        """
        Returns **all** the :class:`~pyccx.core.ElementSet` used and generated in the analysis
        """
        return self._collectSets(setType = ElementSet)

    def getNodeSets(self) -> List[NodeSet]:
        """
        Returns **all** the :class:`pyccx.core.NodeSet` used and generated in the analysis
        """
        return self._collectSets(setType = NodeSet)

    def getSurfaceSets(self) -> List[SurfaceSet]:
        """
        Returns **all** the :class:`pyccx.core.SurfaceSet` used and generated in the analysis
        """
        return self._collectSets(setType=SurfaceSet)

    def writeInput(self) -> str:
        """
        Writes the input deck for the simulation
        """

        self.init()

        self._writeHeaders()
        self._writeMesh()
        self._writeNodeSets()
        self._writeElementSets()
        self._writeKinematicConnectors()
        self._writeMPCs()
        self._writeMaterials()
        self._writeMaterialAssignments()
        self._writeInitialConditions()
        self._writeAnalysisConditions()
        self._writeLoadSteps()

        return self._input

    def _writeHeaders(self):

        self._input += os.linesep
        self._input += '{:*^125}\n'.format(' INCLUDES ')

        for filename in self.includes:
            self._input += '*include,input={:s}'.format(filename)

    def _writeElementSets(self):

        # Collect all sets
        elementSets = self._collectSets(setType = ElementSet)

        if len(elementSets) == 0:
            return

        self._input += os.linesep
        self._input += '{:*^125}\n'.format(' ELEMENT SETS ')

        for elSet in elementSets:
            self._input += os.linesep
            self._input += elSet.writeInput()

    def _writeNodeSets(self):

        # Collect all sets
        nodeSets = self._collectSets(setType=NodeSet)

        if len(nodeSets) == 0:
            return

        self._input += os.linesep
        self._input += '{:*^125}\n'.format(' NODE SETS ')

        for nodeSet in nodeSets:
            self._input += os.linesep
            self._input += nodeSet.writeInput()
            #self._input += '*NSET,NSET={:s}\n'.format(nodeSet['name'])
            #self._input += '*NSET,NSET={:s}\n'.format(nodeSet['name'])
            #self._input += np.array2string(nodeSet['nodes'], precision=2, separator=', ', threshold=9999999999)[1:-1]

    def _writeKinematicConnectors(self):

        if len(self.connectors) < 1:
            return

        self._input += os.linesep
        self._input += '{:*^125}\n'.format(' KINEMATIC CONNECTORS ')

        for connector in self.connectors:

            # A nodeset is automatically created from the name of the connector
            self._input += connector.writeInput()

    def _writeMPCs(self):

        if len(self.mpcSets) < 1:
            return

        self._input += os.linesep
        self._input += '{:*^125}\n'.format(' MPCS ')

        for mpcSet in self.mpcSets:
            self._input += '*EQUATION\n'
            self._input += '{:d}\n'.format(len(mpcSet['numTerms']))  # Assume each line constrains two nodes and one dof
            for mpc in mpcSet['equations']:
                for i in range(len(mpc['eqn'])):
                    self._input += '{:d},{:d},{:d}'.format(mpc['node'][i], mpc['dof'][i], mpc['eqn'][i])

                self._input += os.linesep

    #        *EQUATION
    #        2 # number of terms in equation # typically two
    #        28,2,1.,22,2,-1. # node a id, dof, node b id, dof b

    def _writeMaterialAssignments(self):
        self._input += os.linesep
        self._input += '{:*^125}\n'.format(' MATERIAL ASSIGNMENTS ')

        for matAssignment in self.materialAssignments:
            self._input += '*solid section, elset={:s}, material={:s}\n'.format(matAssignment[0], matAssignment[1])

    def _writeMaterials(self):
        self._input += os.linesep
        self._input += '{:*^125}\n'.format(' MATERIALS ')
        for material in self.materials:
            self._input += material.writeInput()

    def _writeInitialConditions(self):
        self._input += os.linesep
        self._input += '{:*^125}\n'.format(' INITIAL CONDITIONS ')

        for initCond in self.initialConditions:
            self._input += '*INITIAL CONDITIONS,TYPE={:s}\n'.format(initCond['type'].upper())
            self._input += '{:s},{:e}\n'.format(initCond['set'], initCond['value'])
            self._input += os.linesep

        # Write the Physical Constants
        self._input += '*PHYSICAL CONSTANTS,ABSOLUTE ZERO={:e},STEFAN BOLTZMANN={:e}\n'.format(self.TZERO, self.SIGMAB)

    def _writeAnalysisConditions(self):

        self._input += os.linesep
        self._input += '{:*^125}\n'.format(' ANALYSIS CONDITIONS ')

        # Write the Initial Timestep
        self._input += '{:.3f}, {:.3f}\n'.format(self.initialTimeStep, self.defaultTimeStep)

    def _writeLoadSteps(self):

        self._input += os.linesep
        self._input += '{:*^125}\n'.format(' LOAD STEPS ')

        for loadCase in self.loadCases:
            self._input += loadCase.writeInput()

    def _writeMesh(self):

        # TODO make a unique auto-generated name for the mesh
        meshFilename = 'mesh.inp'
        meshPath= os.path.join(self._workingDirectory, meshFilename)

        self.model.writeMesh(meshPath)
        self._input += '*include,input={:s}'.format(meshFilename)

    def checkAnalysis(self) -> bool:
        """
        Routine checks that the analysis has been correctly generated

        :return: bool: True if no analysis error occur
        :raise: AnalysisError: Analysis error that occured
        """

        if len(self.materials) == 0:
            raise AnalysisError('No material models have been assigned to the analysis')

        for material in self.materials:
            if not material.isValid():
                raise AnalysisError('Material ({:s}) is not valid'.format(material.name))


        return True

    def version(self):

        if sys.platform == 'win32':
            cmdPath = os.path.join(self.CALCULIX_PATH, 'ccx.exe ')
            p = subprocess.Popen([cmdPath, '-v'], stdout=subprocess.PIPE, universal_newlines=True )
            stdout, stderr = p.communicate()
            version = re.search(r"(\d+).(\d+)", stdout)
            return int(version.group(1)), int(version.group(2))

        elif sys.platform == 'linux':
            p = subprocess.Popen(['ccx', '-v'], stdout=subprocess.PIPE, universal_newlines=True )
            stdout, stderr = p.communicate()
            version = re.search(r"(\d+).(\d+)", stdout)
            return int(version.group(1)), int(version.group(2))

        else:
            raise NotImplemented(' Platform is not currently supported')

    def results(self) -> ResultProcessor:
        """
        The results obtained after running an analysis
         """
        if self.isAnalysisCompleted():
            return ResultProcessor('input')
        else:
            raise ValueError('Results were not available')

    def isAnalysisCompleted(self) -> bool:
        """ Returns if the analysis was completed successfully """
        return self._analysisCompleted

    def clearAnalysis(self, includeResults: bool = False) -> None:
        """
        Clears any previous files generated from the analysis

        :param includeResults:  If set `True` will also delete the result files generated from the analysis
        """

        filename = 'input' # Base filename for the analysis

        files = [filename + '.inp',
                 filename + '.cvg',
                 filename + '.sta']

        if includeResults:
            files.append(filename + '.frd')
            files.append(filename + '.dat')

        try:
            for file in files:
                filePath = os.path.join(self._workingDirectory,file)
                os.remove(filePath)
        except:
            pass

    def run(self):
        """
        Performs pre-analysis checks on the model and submits the job for Calculix to perform.
        """

        # Reset analysis status
        self._analysisCompleted = False

        print('{:=^60}\n'.format(' RUNNING PRE-ANALYSIS CHECKS '))
        self.checkAnalysis()

        print('{:=^60}\n'.format(' WRITING INPUT FILE '))
        inputDeckContents = self.writeInput()

        inputDeckPath = os.path.join(self._workingDirectory,'input.inp')
        with open(inputDeckPath, "w") as text_file:
            text_file.write(inputDeckContents)

        # Set environment variables for performing multi-threaded
        os.environ["CCX_NPROC_STIFFNESS"] = '{:d}'.format(Simulation.NUMTHREADS)
        os.environ["CCX_NPROC_EQUATION_SOLVER"] = '{:d}'.format(Simulation.NUMTHREADS)
        os.environ["OMP_NUM_THREADS"] = '{:d}'.format(Simulation.NUMTHREADS)

        print('\n{:=^60}\n'.format(' RUNNING CALCULIX '))

        if sys.platform == 'win32':
            cmdPath = os.path.join(self.CALCULIX_PATH, 'ccx.exe ')
            arguments = '-i input'

            cmd = cmdPath + arguments

            popen = subprocess.Popen(cmd, cwd=self._workingDirectory,  stdout=subprocess.PIPE, universal_newlines=True)

            if self.VERBOSE_OUTPUT:
                for stdout_line in iter(popen.stdout.readline, ""):
                    print(stdout_line, end='')

            popen.stdout.close()
            return_code = popen.wait()
            if return_code:
                raise subprocess.CalledProcessError(return_code, cmd)

            # A        :return:nalysis was completed successfully
            self._analysisCompleted = True

        elif sys.platform == 'linux':

            filename = 'input'

            cmdSt = ['ccx', '-i', filename]

            popen = subprocess.Popen(cmdSt, cwd=self._workingDirectory, stdout=subprocess.PIPE, universal_newlines=True)

            if self.VERBOSE_OUTPUT:
                for stdout_line in iter(popen.stdout.readline, ""):
                    print(stdout_line, end='')

            popen.stdout.close()
            return_code = popen.wait()
            if return_code:
                raise subprocess.CalledProcessError(return_code, cmdSt)

            # Analysis was completed successfully
            self._analysisCompleted = True

        else:
            raise NotImplemented(' Platform is not currently supported')
