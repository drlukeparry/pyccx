import re
import os
import sys
import subprocess
import logging

from enum import IntEnum, auto
from typing import Any, List, Type, Optional

import numpy as np

from ..bc import BoundaryCondition
from ..core import Amplitude, Connector, ModelObject, MeshSet, ElementSet, NodeSet, SurfaceSet
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


class AnalysisType(IntEnum):
    """
    The analysis types available in Calculix that may be used for analyses
    """

    STRUCTURAL = auto()
    """ Structural Analysis """

    THERMAL = auto()
    """ Thermal Analysis """

    FLUID = auto()
    """ Fluid Dynamics Analysis"""


class MaterialAssignment(ModelObject):
    """
    MaterialAssignment is a base class for defining the Element Types and :class:`~pyccx.material.Material` that are
    specified for an :class:`~pyccx.core.ElementSet` within the model. These are required to be set for all elements
    that exist within :class:`pyccx.mesh.Mesher` that are defined and exported for use in Calculix.
    """

    def __init__(self, name: str, elementSet: ElementSet, material: Material):

        self._elSet = elementSet
        self._material = material

        super().__init__(name)

    @property
    def material(self) -> Material:
        """ The Material model and parameters assigned to the Material Assignment """
        return self._material

    @material.setter
    def material(self, material: Material) -> None:

        if not isinstance(material, Material):
            raise TypeError('Invalid material assignment provided to MaterialAssignment ({:s})'.format(self.name))

        self._material = material

    @property
    def els(self) -> ElementSet:
        """
        Elements contains the list of Node IDs
        """
        return self._elSet

    @els.setter
    def els(self, elementSet: ElementSet):

        if not isinstance(elementSet, ElementSet):
            raise TypeError('Invalid element set type provided to MaterialAssignment ({:s}).'.format(self.name))

        self._elSet = elementSet

    def writeInput(self) -> str:
        raise Exception('Not implemented')


class SolidMaterialAssignment(MaterialAssignment):
    """
    SolidMaterialAssignment designates elements as solid 3D continuum elements, for the selected elements in a provided
    :class:`~pyccx.core.ElementSet` with the given :class:`Material`. This option should be used for the following class of elements
    including assigning material properties to 3D, plane stress, plane strain and axisymmetric element types. For
    plane stress and plane strain elements the thickness parameter can be specified.
    """
    def __init__(self, name, elementSet: ElementSet, material: Material, thickness: Optional[float] = None):

        self._thickness = thickness
        super().__init__(name, elementSet, material)

    @property
    def thickness(self) -> float:
        return self._thickness

    @thickness.setter
    def thickness(self, thickness: float):

        if thickness is None:
            self._thickness = None
        elif thickness < 1e-8:
            self._thickness = None
        else:
            self._thickness = thickness

    def writeInput(self) -> str:

        outStr = '*solid section, elset={:s}, material={:s}\n'.format(self._elSet.name, self._material.name)

        if self._thickness:
            outStr += '{:e}'.format(self._thickness)

        return outStr


class ShellMaterialAssignment(MaterialAssignment):
    """
    The ShellMaterialAssignment class is used to select shell elements for the selected elements in a provided
    :class:`~pyccx.core.ElementSet` with the given :class:`~pyccx.material.Material`. A thickness must be provided for
    the selected shell elements.
    """

    def __init__(self, name, elementSet: ElementSet, material: Material, thickness: float):

        super().__init__(name, elementSet, material)

        self._thickness = thickness

    @property
    def thickness(self) -> float:
        """
        The thickness of the shell elements

        .. warning::
            The thickness of the shell type should be greater than zero and is required for shell elements.

        .. note::
            The element thickness is constant for the shell assignment
        """

        return self._thickness

    @thickness.setter
    def thickness(self, thickness: float):

        if thickness < 1e-8:
            raise ValueError('The thickness of the shell type should be greater than zero')

        self._thickness = thickness

    def writeInput(self) -> str:
        outStr = '*shell section, elset={:s}, material={:s}\n'.format(self._elSet.name, self._material.name)
        outStr += '{:e}\n'.format(self._thickness)
        return outStr


class Simulation:
    """
    Provides the class for running a Calculix Simulation
    """

    NUMTHREADS: int = 1
    """
    The total number of Threads used by the Calculix Solver
    """

    CALCULIX_PATH: str = ''
    """
    The calculix solver directory path used for Windows platforms. Within the solver directory the executable
    (ccx.exe) must exist and have execution permissions.

    .. note ::
        On Mac OS X, this is the complete path of the executable

    """

    VERBOSE_OUTPUT: bool = True
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
        self._runData = None

    def init(self):

        self._input = ''

    @classmethod
    def setNumThreads(cls, numThreads: int) -> None:
        """
        Sets the number of simulation threads to use in Calculix

        :param numThreads:
        :return:
        """
        cls.NUMTHREADS = numThreads

    @classmethod
    def getNumThreads(cls) -> int:
        """
        Returns the number of threads used by Calculix and GMSH

        :return: int:
        """
        return cls.NUMTHREADS

    @classmethod
    def setCalculixPath(cls, calculixPath: str) -> None:
        """
        Sets the path for the Calculix executable. Necessary when using Windows where there is not a default
        installation procedure for Calculix

        :param calculixPath: Directory containing the Calculix Executable
        """

        if os.path.isdir(calculixPath):
            cls.CALCULIX_PATH = calculixPath

    @classmethod
    def setVerboseOuput(cls, state: bool) -> None:
        """
        Sets if the output from Calculix should be verbose i.e. printed to the console

        :param state: `True` if the output should be printed to the console
        """

        cls.VERBOSE_OUTPUT = state

    def setWorkingDirectory(self, workDir: str) -> None:
        """
        Sets the working directory used during the analysis.

        :param workDir: An accessible working directory path

        """
        if os.path.isdir(workDir) and os.access(workDir, os.W_OK):
            self._workingDirectory = workDir
        else:
            raise ValueError(f"Working directory ({workDir}) is not accessible or writable")

    @property
    def name(self) -> str:
        return self._name

    def getBoundaryConditions(self) -> List[BoundaryCondition]:
        """
        Collects all unique :class:`~pyccx.bc.BoundaryCondition` which are attached
        to each :class:`~pyccx.loadcase.LoadCase` in the analysis

        :return:  All the boundary conditions in the analysis
        """
        bcs = []
        for loadcase in self._loadCases:
            bcs += loadcase.boundaryConditions

        return bcs

    @property
    def loadCases(self) -> List[LoadCase]:
        """
        A list of :class:`~pyccx.loadcase.LoadCase` that have been attached to the analysis
        """
        return self._loadCases

    @loadCases.setter
    def loadCases(self, loadCases: List[LoadCase]) -> None:
        self._loadCases = loadCases

    @property
    def connectors(self) -> List[Connector]:
        """
        List of :class:`~pyccx.core.Connector` used in the analysis
        """
        return self._connectors

    @connectors.setter
    def connectors(self, connectors: List[Connector]) -> None:
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
    def materials(self, materials: List[Material]) -> None:
        self._materials = materials

    @property
    def materialAssignments(self) -> List[MaterialAssignment]:
        """
        Material Assignment applied to a set of elements
        """
        return self._materialAssignments

    @materialAssignments.setter
    def materialAssignments(self, matAssignments: List[MaterialAssignment]) -> None:
        self._materialAssignments = matAssignments

    def _collectAmplitudes(self) -> List[Amplitude]:
        """
        Private function returns a unique set of Element, Nodal, Surface sets which are used by
        the analysis during writing. This reduces the need to explicitly attach them to an analysis.
        """
        amps = {}

        for loadcase in self.loadCases:

            for bc in loadcase.boundaryConditions:
                if bc.amplitude:
                    amps[bc.amplitude.name] = bc.amplitude

        return list(amps.values())

    def _collectSets(self, setType: Optional[Type[MeshSet]] = None) -> Any:
        """
        Private function returns a unique set of Element, Nodal, Surface sets which are used by
        the analysis during writing. This reduces the need to explicitly attach them to an analysis.

        :param setType: The type of Mesh Set to collect
        :return: A list of unique MeshSets obtained for the analysis
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

        for materialAssignment in self.materialAssignments:
            elementSets[materialAssignment.els.name] = materialAssignment.els

        # Iterate through all loadcases and boundary conditions.and find unique values. This is greedy so will override
        # any with same name.
        for loadcase in self.loadCases:

            # Collect result sets node and element sets automatically
            for resultSet in loadcase.resultSet:
                if isinstance(resultSet, ElementResult):
                    elementSets[resultSet.elementSet.name] = resultSet.elementSet
                elif isinstance(resultSet, NodalResult):
                    if resultSet.nodeSet and isinstance(resultSet.nodeSet, NodeSet):
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
    def elementSets(self, val: List[ElementSet]) -> None:
        self._elementSets = val

    @property
    def nodeSets(self) -> List[NodeSet]:
        """
        User-defined :class:`~pyccx.core.NodeSet` manually added to the analysis
        """
        return self._nodeSets

    @nodeSets.setter
    def nodeSets(self, val: List[NodeSet]) -> None:
        self._nodeSets = val

    @property
    def surfaceSets(self) -> List[SurfaceSet]:
        """
        User-defined :class:`pyccx.core.SurfaceSet` manually added to the analysis
        """
        return self._surfaceSets

    @surfaceSets.setter
    def surfaceSets(self, val: List[SurfaceSet]) -> None:
        self._surfaceSets = val

    def getElementSets(self) -> List[ElementSet]:
        """
        Returns **all** the :class:`~pyccx.core.ElementSet` used and generated in the analysis
        """
        return self._collectSets(setType=ElementSet)

    def getNodeSets(self) -> List[NodeSet]:
        """
        Returns **all** the :class:`~pyccx.core.NodeSet` used and generated in the analysis
        """
        return self._collectSets(setType=NodeSet)

    def getSurfaceSets(self) -> List[SurfaceSet]:
        """
        Returns **all** the :class:`~pyccx.core.SurfaceSet` used and generated in the analysis
        """
        return self._collectSets(setType=SurfaceSet)

    def getAmplitudes(self) -> List[Amplitude]:
        """
        Returns *all** the :class:`pyccx.core.Amplitudes` used and generated in the analysis
        """

        return self._collectAmplitudes()

    def writeInput(self) -> str:
        """
        Writes the input deck for the simulation
        """

        self.init()

        self._writeHeaders()

        self._writeMesh()
        logging.info('\t Analysis mesh written to file')
        self._writeNodeSets()
        self._writeElementSets()
        self._writeKinematicConnectors()
        self._writeMPCs()
        self._writeAmplitudes()
        self._writeMaterials()
        self._writeMaterialAssignments()
        self._writeInitialConditions()
        self._writeAnalysisConditions()
        self._writeLoadSteps()

        return self._input

    def _writeAmplitudes(self) -> None:

        amplitudes = self._collectAmplitudes()

        if len(amplitudes) == 0:
            return None

        self._input += '{:*^80}\n'.format(' AMPLITUDES ')

        for amp in amplitudes:
            self._input += amp.writeInput()
            self._input += os.linesep

    def _writeHeaders(self) -> None:

        self._input += '\n'
        self._input += '{:*^125}\n'.format(' INCLUDES ')

        for filename in self.includes:
            self._input += '*include,input={:s}'.format(filename)

    def _writeElementSets(self) -> None:

        # Collect all sets
        elementSets = self._collectSets(setType=ElementSet)

        if len(elementSets) == 0:
            return

        self._input += os.linesep
        self._input += '{:*^125}\n'.format(' ELEMENT SETS ')

        for elSet in elementSets:
            self._input += os.linesep
            self._input += elSet.writeInput()

    def _writeNodeSets(self) -> None:

        # Collect all sets
        nodeSets = self._collectSets(setType=NodeSet)

        if len(nodeSets) == 0:
            return

        self._input += os.linesep
        self._input += '{:*^125}\n'.format(' NODE SETS ')

        for nodeSet in nodeSets:
            self._input += os.linesep
            self._input += nodeSet.writeInput()

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

    def _writeMaterialAssignments(self) -> None:
        self._input += os.linesep
        self._input += '{:*^80}\n'.format(' MATERIAL ASSIGNMENTS ')

        for matAssignment in self.materialAssignments:
            self._input += matAssignment.writeInput()

    def _writeMaterials(self) -> None:
        self._input += os.linesep
        self._input += '{:*^80}\n'.format(' MATERIALS ')
        for material in self.materials:
            self._input += material.writeInput()

    def _writeInitialConditions(self) -> None:
        self._input += os.linesep
        self._input += '{:*^80}\n'.format(' INITIAL CONDITIONS ')

        for initCond in self.initialConditions:
            self._input += '*INITIAL CONDITIONS,TYPE={:s}\n'.format(initCond['type'].upper())
            self._input += '{:s},{:e}\n'.format(initCond['set'], initCond['value'])
            self._input += os.linesep

        # Write the Physical Constants
        self._input += '*PHYSICAL CONSTANTS,ABSOLUTE ZERO={:e},STEFAN BOLTZMANN={:e}\n'.format(self.TZERO, self.SIGMAB)

    def _writeAnalysisConditions(self) -> None:

        self._input += os.linesep
        self._input += '{:*^80}\n'.format(' ANALYSIS CONDITIONS ')

    def _writeLoadSteps(self) -> None:

        self._input += os.linesep
        self._input += '{:*^80}\n'.format(' LOAD STEPS ')

        for loadCase in self.loadCases:
            self._input += loadCase.writeInput()

    def _writeMesh(self) -> None:

        # TODO make a unique auto-generated name for the mesh
        meshFilename = 'mesh.inp'
        meshPath = os.path.join(self._workingDirectory, meshFilename)

        self.model.writeMesh(meshPath)
        self._input += '*include,input={:s}\n'.format(meshFilename)

    def checkAnalysis(self) -> bool:
        """
        Routine checks that the analysis has been correctly generated

        :return: bool: True if no analysis error occur
        :raise: AnalysisError: Analysis error that occurred
        """

        if len(self.materials) == 0:
            raise AnalysisError(self, 'No material models have been assigned to the analysis')

        if len(self.materialAssignments) == 0:
            raise AnalysisError(self, 'No material assignment has been assigned to the analysis')

        for material in self.materials:
            if not material.isValid():
                raise AnalysisError(self, f"Material ({material.name}) is not valid")

        if len(self.model.identifyUnassignedElements()) > 0:
            raise AnalysisError(self, 'Mesh model has unassigned element types')

        return True

    @staticmethod
    def version():

        if sys.platform == 'win32':
            cmdPath = Simulation.CALCULIX_PATH

            # Check executable can be opened and has permissions to be executable
            if not os.path.isfile(cmdPath):
                raise FileNotFoundError(f"Calculix executable not found at path: {cmdPath}")

            # check if the executable is executable
            if not os.access(cmdPath, os.X_OK):
                raise PermissionError(f"Calculix executable at path: {cmdPath} is not executable")

            p = subprocess.Popen([cmdPath, '-v'], stdout=subprocess.PIPE, universal_newlines=True)
            stdout, stderr = p.communicate()
            version = re.search(r"(\d+).(\d+)", stdout)
            return int(version.group(1)), int(version.group(2))

        elif sys.platform == 'linux':

            p = subprocess.Popen(['ccx', '-v'], stdout=subprocess.PIPE, universal_newlines=True)
            stdout, stderr = p.communicate()
            version = re.search(r"(\d+).(\d+)", stdout)
            return int(version.group(1)), int(version.group(2))

        elif sys.platform == 'darwin':

            # Check executable can be opened and has permissions to be executable
            if not os.path.isfile(Simulation.CALCULIX_PATH):
                raise FileNotFoundError(f"Calculix executable not found at path: {Simulation.CALCULIX_PATH}")

            # check if the executable is executable
            if not os.access(Simulation.CALCULIX_PATH, os.X_OK):
                raise PermissionError(f"Calculix executable at path: {Simulation.CALCULIX_PATH} is not executable")

            p = subprocess.Popen([Simulation.CALCULIX_PATH, '-v'], stdout=subprocess.PIPE, universal_newlines=True)
            stdout, stderr = p.communicate()
            version = re.search(r"(\d+).(\d+)", stdout)
            return int(version.group(1)), int(version.group(2))
        else:
            raise NotImplementedError(' Platform is not currently supported')

    def results(self) -> ResultProcessor:
        """
        The results obtained after running an analysis
         """

        workingResultsPath = os.path.join(self._workingDirectory, 'input')

        if self.isAnalysisCompleted():
            return ResultProcessor(workingResultsPath)
        else:
            raise RuntimeError('Results were not available')

    def isAnalysisCompleted(self) -> bool:
        """ Returns ``True`` if the analysis was completed successfully """
        return self._analysisCompleted

    def clearAnalysis(self, includeResults: Optional[bool] = False) -> None:
        """
        Clears any previous files generated from the analysis

        :param includeResults:  If set `True` will also delete the result files generated from the analysis
        """

        filename = 'input'   # Base filename for the analysis

        files = [filename + '.inp',
                 filename + '.cvg',
                 filename + '.sta']

        if includeResults:
            files.append(filename + '.frd')
            files.append(filename + '.dat')

        try:
            for file in files:
                filePath = os.path.join(self._workingDirectory, file)
                os.remove(filePath)
        except:
            pass

    def monitor(self, filename: str):

        # load the .sta file for convegence monitoring

        staFilename = '{:s}.sta'.format(filename)

        """
        Note:
        Format of each row in the .sta file corresponds with
        0 STEP
        1 INC
        2 ATT
        3 ITRS
        4 TOT TIME
        5 STEP TIME
        6 INC TIME
        """
        with open(staFilename, 'r') as f:

            # check the first two lines of the .sta file are correct
            line1 = f.readline()
            line2 = f.readline()

            if not('SUMMARY OF JOB INFORMATION' in line1 and 'STEP' in line2):
                raise Exception('Invalid .sta file generated')

            line = f.readline()

            convergenceOutput = []

            while line:
                out = re.search(r"\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)*", line)

                if out:
                    out = [float(val) for val in out.groups()]
                    convergenceOutput.append(out)

                line = f.readline()

        convergenceOutput = np.array(convergenceOutput)

        cvgFilename = f"{filename}.cvg"

        """
        Note:

        Format of the CVF format consists of the following parameters
        0 STEP
        1 INC
        2 ATT
        3 ITER
        4 CONT EL
        5 RESID FORCE
        6 CORR DISP
        7 RESID FLUX
        8 CORR TEMP
        """
        with open(cvgFilename, 'r') as f:

            # check the first two lines of the .sta file are correct
            line1 = f.readline()
            line2 = f.readline()
            line3 = f.readline()
            line4 = f.readline()

            if not ('SUMMARY OF C0NVERGENCE INFORMATION' in line1 and
                    'STEP' in line2):
                raise Exception('Invalid .cvg file generated')

            line = f.readline()

            convergenceOutput2 = []

            while line:
                out = re.search(r"\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)*", line)

                if out:
                    out = [float(val) for val in out.groups()]
                    convergenceOutput2.append(out)

                line = f.readline()

            convergenceOutput2 = np.array(convergenceOutput2)

        return convergenceOutput, convergenceOutput2

    def checkLine(self, line):

        self._runData = {}

        if 'Total CalculiX Time:' in line:
            runTime = re.search(r"Total CalculiX Time: (\S*)", line)[1]
            runTime = float(runTime)

            self._runData['runTime'] = runTime

    def run(self):
        """
        Performs pre-analysis checks on the model and submits the job for Calculix to perform.
        """

        # Reset analysis status
        self._analysisCompleted = False

        logging.info('{:=^60}'.format(' RUNNING PRE-ANALYSIS CHECKS '))
        if self.checkAnalysis():
            logging.info('\t Analysis checks were successfully completed')

        logging.info('{:=^60}'.format(' WRITING ANALYSIS INPUT FILE '))
        inputDeckContents = self.writeInput()

        logging.info('\t Analysis input file has been generated')
        inputDeckPath = os.path.join(self._workingDirectory, 'input.inp')

        with open(inputDeckPath, "w") as text_file:
            text_file.write(inputDeckContents)

        logging.info('\t Analysis input file ({:s}) has been written to file'.format(inputDeckPath))

        # Set environment variables for performing multi-threaded
        os.environ["CCX_NPROC_STIFFNESS"] = '{:d}'.format(Simulation.NUMTHREADS)
        os.environ["CCX_NPROC_EQUATION_SOLVER"] = '{:d}'.format(Simulation.NUMTHREADS)
        os.environ["NUMBER_OF_PROCESSORS"] = '{:d}'.format(Simulation.NUMTHREADS)
        os.environ["OMP_NUM_THREADS"] = '{:d}'.format(Simulation.NUMTHREADS)

        logging.info('{:=^60}'.format(' RUNNING CALCULIX '))

        if sys.platform == 'win32':
            cmdPath = self.CALCULIX_PATH

            # Check executable can be opened and has permissions to be executable
            if not os.path.isfile(cmdPath):
                raise FileNotFoundError(f"Calculix executable not found at path: {cmdPath}")

            # check if the executable is executable
            if not os.access(cmdPath, os.X_OK):
                raise PermissionError(f"Calculix executable at path: {cmdPath} is not executable")

            arguments = '-i input'

            cmd = cmdPath + arguments

            popen = subprocess.Popen(cmd, cwd=self._workingDirectory,  stdout=subprocess.PIPE, universal_newlines=True)

            for stdout_line in iter(popen.stdout.readline, ""):

                if not stdout_line or stdout_line == '\n':
                    continue

                if "Using up to " in stdout_line:
                    continue

                if Simulation.VERBOSE_OUTPUT:
                    print(stdout_line, end='')

                self.checkLine(stdout_line)

            popen.stdout.close()
            return_code = popen.wait()
            if return_code:
                raise subprocess.CalledProcessError(return_code, cmd)

            # Analysis was completed successfully
            self._analysisCompleted = True

        elif sys.platform == 'linux':

            filename = 'input'

            cmdSt = ['ccx', '-i', filename]

            popen = subprocess.Popen(cmdSt, cwd=self._workingDirectory,
                                     stdout=subprocess.PIPE,
                                     universal_newlines=True)

            for stdout_line in iter(popen.stdout.readline, ""):

                if not stdout_line or stdout_line == '\n':
                    continue

                if "Using up to " in stdout_line:
                    continue

                if Simulation.VERBOSE_OUTPUT:
                    print(stdout_line, end='')
                self.checkLine(stdout_line)

            popen.stdout.close()
            return_code = popen.wait()
            if return_code:
                raise subprocess.CalledProcessError(return_code, cmdSt)

            # Analysis was completed successfully
            self._analysisCompleted = True

        elif sys.platform == 'darwin':

            filename = 'input'

            # Check executable can be opened and has permissions to be executable
            if not os.path.isfile(Simulation.CALCULIX_PATH):
                raise FileNotFoundError(f"Calculix executable not found at path: {Simulation.CALCULIX_PATH}")

            # check if the executable is executable
            if not os.access(Simulation.CALCULIX_PATH, os.X_OK):
                raise PermissionError(f"Calculix executable at path: {Simulation.CALCULIX_PATH} is not executable")

            cmdSt = [self.CALCULIX_PATH, '-i', filename]

            popen = subprocess.Popen(cmdSt, cwd=self._workingDirectory,
                                     stdout=subprocess.PIPE,
                                     universal_newlines=True)

            for stdout_line in iter(popen.stdout.readline, ""):

                if not stdout_line or stdout_line == '\n':
                    continue

                if "Using up to " in stdout_line:
                    continue

                if Simulation.VERBOSE_OUTPUT:
                    print(stdout_line, end='')

                self.checkLine(stdout_line)

            popen.stdout.close()
            return_code = popen.wait()

            if return_code:
                raise subprocess.CalledProcessError(return_code, cmdSt)

            # Analysis was completed successfully
            self._analysisCompleted = True

        else:
            raise NotImplementedError(' Platform is not currently supported')
