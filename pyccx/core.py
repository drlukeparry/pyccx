# -*- coding: utf-8 -*-


import re  # used to get info from frd file
import os
import sys
import subprocess  # used to check ccx version
from enum import Enum, auto
from typing import List, Tuple, Type
import logging

from .boundarycondition import BoundaryCondition
from .loadcase import LoadCase
from .material import Material
from .mesh import Mesher
from .results import ElementResult, NodalResult, ResultProcessor

import numpy as np


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
    STRUCTURAL = auto()
    THERMAL = auto()
    FLUID = auto()


class MeshSet:

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
        self._nodes = nodes

    @property
    def nodes(self):
        """
        Nodes contains the list of Node IDs
        """
        return self._nodes

    @nodes.setter
    def nodes(self, nodes):
        self._nodes = nodes

    def writeInput(self) -> str:
        out = '*NSET,NSET={:s}\n'.format(self.name)
        out += np.array2string(self.nodes, precision=2, separator=', ', threshold=9999999999)[1:-1]
        return out


class ElementSet(MeshSet):
    """
    An element set is basic entity for storing element set lists.The set remains constant without any dynamic referencing
     to any underlying geometric entities.
    """
    def __init__(self, name, els):
        super().__init__(name)
        self._els = els

    @property
    def els(self):
        """
        Elements contains the list of Node IDs
        """
        return self._els

    @els.setter
    def els(self, elements):
        self._els = elements

    def writeInput(self) -> str:

        out = '*ELSET,ELSET={:s\n}'.format(self.name)
        out += np.array2string(self.els, precision=2, separator=', ', threshold=9999999999)[1:-1]
        return out


class SurfaceSet(MeshSet):
    """
    A surface-set set is basic entity for storing element face lists, typically for setting directional fluxes onto
    surface elements based on the element ordering. The set remains constant without any dynamic referencing
     to any underlying geometric entities.
    """
    def __init__(self, name, surfacePairs):

        super().__init__(name)
        self._elSurfacePairs = surfacePairs

    @property
    def surfacePairs(self):
        """
        Elements with the associated face orientations are specified as Nx2 numpy array, with the first column being
        the element Id, and the second column the chosen face orientation
        """
        return self._elSurfacePairs

    @surfacePairs.setter
    def surfacePairs(self, surfacePairs):
        self._elSurfacePairs = surfacePairs

    def writeInput(self) -> str:

        out = '*SURFACE,NAME={:s}\n'.format(self.name)

        for i in range(self._elSurfacePairs.shape[0]):
            out += '{:d},S{:d}\n'.format(self._elSurfacePairs[i,0], self._elSurfacePairs[i,1])

        #out += np.array2string(self.els, precision=2, separator=', ', threshold=9999999999)[1:-1]
        return out


class Connector:
    """
     A Connector ir a rigid connector between a set of nodes and an (optional) reference node.
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
            raise ValueError('Invalid type for nodes passed to Connector()')

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
    UX = 1
    UY = 2
    UZ = 3
    RX = 4
    RY = 5
    RZ = 6
    T = 11

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

    def setWorkingDirectory(self, workDir):
        if os.path.isdir(workDir) and os.access(workDir, os.W_OK):
            self._workingDirectory = workDir
        else:
            raise ValueError('Working directory ({:s}) is not accessible or writable'.format(workDir))

    @property
    def name(self):
        return self._name

    def getBoundaryConditions(self) -> List[BoundaryCondition]:
        """
        Collects all boundary conditions which are attached to loadcases in the analysis
        """
        bcs = []
        for loadcase in self._loadCases:
            bcs += loadcase.boundaryConditions

        return bcs

    @property
    def loadCases(self) -> List[LoadCase]:
        """
        The Loadcases for the analysis
        """
        return self._loadCases

    @loadCases.setter
    def loadCases(self, loadCases: List[LoadCase]):
        self._loadCases = loadCases

    @property
    def connectors(self) -> List[Connector]:
        """
        List of connectors used in the simulation
        """
        return self._connectors

    @connectors.setter
    def connectors(self, connectors):
        self._connectors = connectors

    @property
    def mpcSets(self):
        return self._mpcSets

    @mpcSets.setter
    def mpcSets(self, value):
        self._mpcSets = value

    @property
    def materials(self) -> List[Material]:
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
        :return:
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
                    elementSets[resultSet.elSet.name] = resultSet.elSet
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
        User-defined element sets manually added to the analysis
        """
        return self._elementSets

    @elementSets.setter
    def elementSets(self, val = List[ElementSet]):
        """
        User-defined element sets manually added to the analysis
        """
        self._elementSets = val

    @property
    def nodeSets(self) -> List[NodeSet]:
        """
        User-defined node sets manually added to the analysis
        """
        return self._nodeSets

    @nodeSets.setter
    def nodeSets(self, val=List[NodeSet]):
        """
        User-defined element sets manually added to the analysis
        """
        nodeSets = val

    @property
    def surfaceSets(self) -> List[SurfaceSet]:
        """
        User-defined element sets manually added to the analysis
        """
        return self._nodeSets

    @surfaceSets.setter
    def surfaceSets(self, val=List[SurfaceSet]):
        """
        User-defined element sets manually added to the analysis
        """
        surfaceSets = val

    def getElementSets(self) -> List[ElementSet]:
        """
        Returns all the element sets used and generated in the analysis
        """
        return self._collectSets(setType = ElementSet)

    def getNodeSets(self) -> List[NodeSet]:
        """
        Returns all the element sets used and generated in the analysis
        """
        return self._collectSets(setType = NodeSet)

    def getSurfaceSets(self) -> List[SurfaceSet]:
        """
        Returns all the element sets used and generated in the analysis
        """
        return self._collectSets(setType=SurfaceSet)

    def writeHeaders(self):

        self._input += os.linesep
        self._input += '{:*^125}\n'.format(' INCLUDES ')

        for filename in self.includes:
            self._input += '*include,input={:s}'.format(filename)

    def writeInput(self) -> str:
        """
        Writes the input deck for the simulation
        """

        self.init()

        self.writeHeaders()
        self.writeMesh()
        self.writeNodeSets()
        self.writeElementSets()
        self.writeKinematicConnectors()
        self.writeMPCs()
        self.writeMaterials()
        self.writeMaterialAssignments()
        self.writeInitialConditions()
        self.writeAnalysisConditions()
        self.writeLoadSteps()

        return self._input


    def writeElementSets(self):
        """
        Functions writes element sets
        """

        # Collect all sets
        elementSets = self._collectSets(setType = ElementSet)

        if len(elementSets) == 0:
            return

        self._input += os.linesep
        self._input += '{:*^125}\n'.format(' ELEMENT SETS ')

        for elSet in self.elementSets:
            self._input += os.linesep
            self._input += elSet.writeInput()

    def writeNodeSets(self):

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

    def writeKinematicConnectors(self):

        if len(self.connectors) < 1:
            return

        self._input += os.linesep
        self._input += '{:*^125}\n'.format(' KINEMATIC CONNECTORS ')

        for connector in self.connectors:

            # A nodeset is automatically created from the name of the connector
            self._input += connector.writeInput()

    def writeMPCs(self):

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

    def writeMaterialAssignments(self):
        self._input += os.linesep
        self._input += '{:*^125}\n'.format(' MATERIAL ASSIGNMENTS ')

        for matAssignment in self.materialAssignments:
            self._input += '*solid section, elset={:s}, material={:s}\n'.format(matAssignment[0], matAssignment[1])

    def writeMaterials(self):
        self._input += os.linesep
        self._input += '{:*^125}\n'.format(' MATERIALS ')
        for material in self.materials:
            self._input += material.writeInput()

    def writeInitialConditions(self):
        self._input += os.linesep
        self._input += '{:*^125}\n'.format(' INITIAL CONDITIONS ')

        for initCond in self.initialConditions:
            self._input += '*INITIAL CONDITIONS,TYPE={:s}\n'.format(initCond['type'].upper())
            self._input += '{:s},{:e}\n'.format(initCond['set'], initCond['value'])
            self._input += os.linesep

        # Write the Physical Constants
        self._input += '*PHYSICAL CONSTANTS,ABSOLUTE ZERO={:e},STEFAN BOLTZMANN={:e}\n'.format(self.TZERO, self.SIGMAB)

    def writeAnalysisConditions(self):

        self._input += os.linesep
        self._input += '{:*^125}\n'.format(' ANALYSIS CONDITIONS ')

        # Write the Initial Timestep
        self._input += '{:.3f}, {:.3f}\n'.format(self.initialTimeStep, self.defaultTimeStep)

    def writeLoadSteps(self):

        self._input += os.linesep
        self._input += '{:*^125}\n'.format(' LOAD STEPS ')

        for loadCase in self.loadCases:
            self._input += loadCase.writeInput()

    def writeMesh(self):

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
        """ Returns the results obtained after running an analysis """
        if self.isAnalysisCompleted():
            return ResultProcessor('input')
        else:
            raise ValueError('Results were not available')

    def isAnalysisCompleted(self) -> bool:
        """ Returns if the analysis was completed successfully. """
        return self._analysisCompleted

    def clearAnalysis(self, includeResults:bool = False) -> None:
        """ Clears any files generated from the analysis

        :param includeResults:  If set True will also delete the result files generated from the analysis
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

            # Analysis was completed successfully
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
