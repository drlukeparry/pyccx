# -*- coding: utf-8 -*-


import re  # used to get info from frd file
import os
import sys
import subprocess  # used to check ccx version
from enum import Enum, auto
from typing import List, Tuple
import logging

from .mesh import Mesher

import gmsh
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


class NodeSet:
    """
     An node set is basic entity for storing node set lists. The set remains constant without any dynamic referencing
     to any underlying geometric entities.
     """
    def __init__(self, name, nodes):
        self.name = name
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


class ElementSet:
    """
    An element set is basic entity for storing element set lists.The set remains constant without any dynamic referencing
     to any underlying geometric entities.
    """
    def __init__(self, name, els):
        self.name =  name
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


class SurfaceSet:
    """
    A surface-set set is basic entity for storing element face lists, typically for setting directional fluxes onto
    surface elements based on the element ordering. The set remains constant without any dynamic referencing
     to any underlying geometric entities.
    """
    def __init__(self, name, surfacePairs):
        self.name =  name
        self._elSurfacePairs = surfacePairs

    @property
    def surfacePairs(self):
        """
        Elements with the associated face orientations are specified as Nx2 numpy array, with the first column being
        the element Id, and the second column the chosen face orientation
        """
        return self._els

    @surfacePairs.setter
    def els(self, surfacePairs):
        self._elSurfacePairs = surfacePairs

    def writeInput(self) -> str:

        out = '*SURFACE,NAME={:s}\n'.format(self.name)

        for i in range(self._elSurfacePairs.shape[0]):
            out += '{:d},S{:d}\n'.format(self._elSurfacePairs[i,0], self._elSurfacePairs[i,1])

        out += np.array2string(self.els, precision=2, separator=', ', threshold=9999999999)[1:-1]
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
        if isinstance(self.redNode, int):
            strOut += ',REF NODE={:d}\n'.format(self.refNode)
        else:
            strOut += '\n'

        return strOut

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

        self.mpcSets = []
        self.connectors = []
        self.materials = []
        self.materialAssignments = []
        self.model = meshModel

        self.initialTimeStep = 0.1
        self.defaultTimeStep = 0.1
        self.totalTime = 1.0
        self.useSteadyStateAnalysis = True

        self.TZERO = -273.15
        self.SIGMAB = 5.669E-8
        self._numThreads = 1

        self.initialConditions = []  # 'dict of node set names,
        self.loadCases = []

        self._nodeSets = []
        self._elSets = []

        self.nodeSets = []
        self.elSets = []
        self.includes = []

    def init(self):

        self._input = ''
        self._nodeSets = self.nodeSets
        self._elSets = self.elSets

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

    def writeHeaders(self):

        self._input += os.linesep
        self._input += '{:*^125}\n'.format(' INCLUDES ')

        for filename in self.includes:
            self._input += '*include,input={:s}'.format(filename)

    def prepareConnectors(self):
        """
        Creates node sets for any RBE connectors used in the simulation
        """
        # Kinematic Connectors require creating node sets
        # These are created and added to the node set collection prior to writing

        numConnectors = 1

        for connector in self.connectors:
            # Node are created and are an attribute of a Connector
            self._nodeSets.append(connector.nodeset)

            numConnectors += 1

    def writeInput(self) -> str:
        """
        Writes the input deck for the simulation
        """

        self.init()

        self.prepareConnectors()

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

        if len(self._elSets) == 0:
            return

        self._input += os.linesep
        self._input += '{:*^125}\n'.format(' ELEMENT SETS ')

        for elSet in self._elSets:
            self._input += os.linesep
            self._input += elSet.writeInput()

            #self._input += '*ELSET,ELSET={:s\n}'.format(elSet['name'])
            #self._input += np.array2string(elSet['els'], precision=2, separator=', ', threshold=9999999999)[1:-1]

    def writeNodeSets(self):

        if len(self._nodeSets) == 0:
            return

        self._input += os.linesep
        self._input += '{:*^125}\n'.format(' NODE SETS ')

        for nodeSet in self._nodeSets:
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
            self.input += connector.writeInput()

    def writeMPCs(self):

        if len(self.mpcSets) < 1:
            return

        self._input += os.linesep
        self._input += '{:*^125}\n'.format(' MPCS ')

        for mpcSet in self.mpcSets:
            self.input += '*EQUATION\n'
            self.input += '{:d}\n'.format(len(mpcSet['numTerms']))  # Assume each line constrains two nodes and one dof
            for mpc in mpcSet['equations']:
                for i in range(len(mpc['eqn'])):
                    self._input += '{:d},{:d},{:d}'.format(mpc['node'][i], mpc['dof'][i], mpc['eqn'][i])

                self.input += os.linesep

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

    def isAnalysisCompleted(self) -> bool:
        """ Returns whether the analysis was completed successfully. """
        return self._analysisCompleted

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
