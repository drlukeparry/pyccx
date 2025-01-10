from typing import Dict, List, Optional, Tuple, Union

import abc
import re
import os
import logging
import numpy as np

from ..core import ElementSet, NodeSet


class ResultsValue:
    """
    The following class attributes are available for post-processing the result file and are used as an enumeration
    for selecting the desired results to be saved to the .frd and .dat file processed in the :class:`ResultProcessor`.
    """

    # Nodal Quantities
    DISP = 'disp'
    """ Nodal Displacement Component  """

    STRESS = 'stress'
    """ Nodal Cauchy Stress Components """

    VMSTRESS = 'stressVM'
    """ Nodal von Mises Stress """

    STRAIN = 'strain'
    """ Nodal Strain tensor components """

    FORCE = 'force'
    """ Nodal Reaction Force Components """

    TEMP = 'temp'
    """ Nodal Temperature """

    # Elemental  Quantities
    ELSTRESS = 'elStress'
    ELHEATFLUX = 'elHeatFlux'


class Result(abc.ABC):
    """
    Base Class for all Calculix Results
    """
    def __init__(self):
        self._frequency = 1

    @property
    def frequency(self):
        """
        The frequency for storing results sections in Calculix
        """
        return self._frequency

    @frequency.setter
    def frequency(self, frequency: int):
        self._frequency = frequency

    @abc.abstractmethod
    def writeInput(self):
        raise NotImplementedError()


class NodalResult(Result):
    """
    The NodalResult when attached to a :class:`~pyccx.loadcase.LoadCase` will inform Calculix to save the nodal
    values onto the file (.frd) for the selected nodes attached in the specified :class:`NodeSet`.
    """

    def __init__(self, nodeSet: Union[str, NodeSet] = None):

        if nodeSet and not (isinstance(nodeSet, NodeSet) or isinstance(nodeSet, str)):
            raise TypeError('NodalResult must be initialized with a NodeSet object or name of set')

        self._nodeSet = nodeSet
        self._displacement = True
        self._temperature = False
        self._reactionForce = False
        self._heatFlux = False
        self._cauchyStress = False  # Int points are interpolated to nodes
        self._plasticStrain = False
        self._strain = False

        self._expandShellElements = False

        super().__init__()

    @property
    def displacement(self) -> bool:
        """ Include the nodal displacement components in the results """
        return self._displacement

    @displacement.setter
    def displacement(self, state: bool) -> None:
        self._displacement = state

    @property
    def temperature(self) -> bool:
        """
        Include the nodal temperature in the results
        """
        return self._temperature

    @temperature.setter
    def temperature(self, state: bool) -> None:
        self._temperature = state

    @property
    def reactionForce(self) -> bool:
        """
        Include the nodal reaction forces in the results
        """
        return self._reactionForce

    @reactionForce.setter
    def reactionForce(self, state: bool) -> None:
        self._reactionForce = state

    @property
    def heatFlux(self) -> bool:
        """
        Include the nodal heat flux components in the results
        """
        return self._heatFlux

    @heatFlux.setter
    def heatFlux(self, state) -> None:
        self._heatFlux = state

    @property
    def cauchyStress(self) -> bool:
        """
        Include the extrapolated nodal cauchy stress components in the results
        """
        return self._cauchyStress

    @cauchyStress.setter
    def cauchyStress(self, state) -> None:
        self._cauchyStress = state

    @property
    def strain(self) -> bool:
        """
        Include the strain components in the results
        """
        return self._strain

    @strain.setter
    def strain(self, state) -> None:
        self._strain = state

    @property
    def plasticStrain(self) -> bool:
        """
        Include equivalent plastic strain variable in the results
        """
        return self._plasticStrain

    @plasticStrain.setter
    def plasticStrain(self, state) -> None:
        self._plasticStrain = state

    @property
    def expandShellElements(self) -> bool:
        """
        Setting this property will instruct calculix to export the  node values that are obtained implicitly when the
        nodes from the shell elements are expanded/project rom their mid-surface region. This is useful for
        post-processing and representing an equivalent volumetric mesh for the shell elements.
        """
        return self._expandShellElements

    @expandShellElements.setter
    def expandShellElements(self, state: bool):
        self._expandShellElements = state

    @property
    def nodeSet(self) -> Union[NodeSet, str]:
        """
        The :class:`NodeSet` to obtain values for post-processing.
        """
        return self._nodeSet

    @nodeSet.setter
    def nodeSet(self, nodeSet: Union[NodeSet, str]):

        if not (isinstance(nodeSet, NodeSet) or isinstance(nodeSet, str)):
            raise TypeError('NodalResult nodeset must be a NodeSet object or name of set')

        self._nodeSet = nodeSet

    def writeInput(self):
        inputStr = ''

        if self._nodeSet:
            """ Obtain the selected nodal quantities for the model using *NODE PRINT option """
            inputStr += '*NODE PRINT'

            if isinstance(self._nodeSet, str) and self._nodeSet != '':

                inputStr += ', NSET={:s}, '.format(self.nodeSet)

            if isinstance(self._nodeSet, NodeSet):
                inputStr += ', NSET={:s} '.format(self.nodeSet.name)

        else:
            """ Obtain the entire nodal quantities for the model using *NODE FILE option """
            inputStr += '*NODE FILE'

            if not self._expandShellElements:
                inputStr += ', OUTPUT=2D '

        inputStr += ', FREQUENCY={:d}\n'.format(self._frequency)

        lineStr = []

        if self._displacement:
            lineStr.append('U')

        if self._temperature:
            lineStr.append('NT')

        if self._reactionForce:
            lineStr.append('RF')

        inputStr += ', '.join(lineStr)

        inputStr += '\n'
        inputStr += self.writeElementInput()

        return inputStr

    def writeElementInput(self):

        lineStr = []

        if self._cauchyStress:
            lineStr.append('S')

        if self._strain:
            lineStr.append('E')

        if self._plasticStrain:
            lineStr.append('PEEQ')

        if self._heatFlux:
            lineStr.append('HFL')

        if len(lineStr) == 0:
            return ''

        elInputStr = '*EL FILE'

        if self._nodeSet:

            if isinstance(self._nodeSet, NodeSet):
                elInputStr += ', NSET={:s}'.format(self._nodeSet.name)
            else:
                elInputStr += ', NSET={:s}'.format(self._nodeSet)

        elInputStr += ', FREQUENCY={:d}\n'.format(self._frequency)

        elInputStr += ', '.join(lineStr)
        elInputStr += '\n'

        return elInputStr


class ElementResult(Result):
    """
    Including an :class:`ElementResult` in a :class:`~pyccx.loadcase.LoadCase` will inform Calcuix to save the
    elemental integration properties to the (.dat) file for the selected :class:`ElementSet` in the chosen working
    directory specified in the :class:`~pyccx.analysis.Simulation`.
    """
    def __init__(self, elSet: ElementSet):

        if not (isinstance(elSet, ElementSet) or isinstance(elSet, str)):
            raise TypeError('ElementResult must be initialized with an ElementSet object or string')

        self._elSet = elSet
        self._strain = False
        self._mechanicalStrain = False
        self._cauchyStress = False
        self._plasticStrain = False
        self._heatFlux = False
        self._ESE = False

        super().__init__()

    @property
    def plasticStrain(self) -> bool:
        """
        The equivalent plastic strain
        """
        return self._plasticStrain

    @plasticStrain.setter
    def plasticStrain(self, state: bool) -> None:
        self._plasticStrain = state

    @property
    def strain(self) -> bool:
        """
        The total Lagrangian strain for (hyper)-elastic materials and incremental plasticity
        and the total Eulerian strain for deformation plasticity.
        """
        return self._strain

    @strain.setter
    def strain(self, state: bool) -> None:
        self._strain = state

    @property
    def mechanicalStrain(self) -> bool:
        """
        This is the mechanical Lagrangian strain for (hyper)elastic materials and incremental plasticity and the
        mechanical Eulerian strain for deformation plasticity (mechanical strain = total strain - thermal strain).
        """
        return self._mechanicalStrain

    @mechanicalStrain.setter
    def mechanicalStrain(self, state: bool) -> None:
        self._mechanicalStrain = state

    @property
    def ESE(self) -> bool:
        """
        Obtain the internal strain energy per unit volume
        """
        return self._ESE

    @ESE.setter
    def ESE(self, state: bool) -> None:
        """
        The internal energy per unit volume
        """
        self._ESE = state

    @property
    def heatFlux(self) -> bool:
        """
        Include heat flux in the output
        """
        return self._heatFlux

    @heatFlux.setter
    def heatFlux(self, state) -> None:
        self._heatFlux = state

    @property
    def cauchyStress(self) -> bool:
        """
        Include cauchy stress components in the output
        """
        return self._cauchyStress

    @cauchyStress.setter
    def cauchyStress(self, state) -> None:
        self._cauchyStress = state

    @property
    def elementSet(self) -> Union[ElementSet, str]:
        """
        The elementset to obtain values for post-processing.
        """

        return self._elSet

    @elementSet.setter
    def elementSet(self, elSet: ElementSet) -> None:

        if not (isinstance(elSet, NodeSet) or isinstance(elSet, str)):
            raise TypeError('ElementResult must be initialised with a NodeSet object or name of set')

        self._elSet = elSet

    def writeInput(self) -> str:

        outStr = ''

        elName = self._elSet.name if isinstance(self._elSet, ElementSet) else self._elSet

        outStr += '*EL PRINT, ELSET={:s}, FREQUENCY={:d}\n'.format(elName, self._frequency)

        if self._cauchyStress:
            outStr += 'S\n'

        if self._strain:
            outStr += 'E\n'

        if self._mechanicalStrain:
            outStr += 'ME\n'

        if self._plasticStrain:
            outStr += 'PEEQ\n'

        if self._ESE:
            outStr += 'ELSE\n'

        if self._heatFlux:
            outStr += 'HFL\n'

        return outStr


class ResultProcessor:
    """
    ResultProcessor takes the output (results) file from the Calculix simulation, specified in the working directory
    for the :class:`~pyccx.analysis.Simulation`. The class processes the ASCII .frd file to load the results into a
    structure.


    The typical usage is that the :class:`~pyccx.analysis.Simulation` will be successfully run and provide a
    convenience handle to process the results:

    .. code-block:: python

        # Run the analysis object
        analysis = pyccx.analysis.Simulation()

        # ...
        analysis.run()

        # Open the results  file ('input') is currently the file that is generated by PyCCX
        results = analysis.results()

        # The call to read must be done to load all loadcases and timesteps from the results file
        results.read()

    Individual timesteps (increments) are seperated and may be accessed accordingly using the
    :attr:`~ResultsProcessor.increments` property. The results are stored in a dictionary with the increment index
    and a corresponding dictionary of results. The results dictionary contains the nodal and elemental results that
    are specified previously in the analysis.
    """

    def __init__(self, jobName: str):

        self._increments = {}
        self.jobName = jobName

        self._elements = None
        self._nodes = None

        logging.info(f"Results file prefix set to {jobName}")

    @property
    def increments(self) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Stored increments of the Calculix results file
        """
        return self._increments

    @property
    def nodes(self) -> Tuple[List[int], List[List[int]]]:
        """
        Nodes identified in the Calculix results file
        """

        nIds = self._nodes[:, 0]
        nCoords = self._nodes[:, 1:].reshape(-1, 3)

        return nIds, nCoords

    @property
    def elements(self) -> Tuple[List[int], List[int], List[List[int]]]:
        """
        Elements identified in the Calculix results file

        A tuple of element ids, element types and element connectivity are returned
        """
        elIds = [el[0] for el in self._elements]
        elTypes = [el[1] for el in self._elements]
        elCon = [el[2:] for el in self._elements]

        return elIds, elTypes, elCon

    def getNodeResult(self, increment: dict,
                      resultKey: ResultsValue,
                      nodeIds: Optional[np.array] = None) -> Tuple[np.array, np.array]:
        """
        Returns a nodal result at step increment, for a corresponding :class:`ResultsValue` type. The nodeIds parameter
        is optional.

        :param increment: The selected increment index available
        :param resultKey: A Valid Nodal Quantity in ResultsValue
        :param nodeIds: A list  of node ids
        :return: A tuple of node ids and corresponding result values

        :raises: Exception if the increment or result key does not exist
        """
        if not self.hasResults():
            raise Exception('No results have been read')

        resultsIncrement = self._increments.get(increment, None)

        if resultsIncrement is None:
            raise Exception(f"Increment {increment} does not exist")

        validNodeKeys = [ResultsValue.DISP, ResultsValue.STRESS, ResultsValue.VMSTRESS, ResultsValue.STRAIN,
                         ResultsValue.FORCE, ResultsValue.TEMP]

        if resultKey not in validNodeKeys:
            raise Exception('Invalid result key specified - not a Nodal Result ')

        result = resultsIncrement.get(resultKey, None)

        if result is None:
            raise Exception('Result {:s} does not exist in increment {:d}'.format(resultKey, id))

        if nodeIds is None:
            # Return the results including the nodal ids
            return result[:, 0], result[:, 1:]
        else:
            if isinstance(nodeIds, NodeSet):
                nIds = nodeIds.nodes  # Convert FEA ids to Pythonic indexing
            else:
                nIds = nodeIds

            # Perform a set intersection
            fndIds = np.argwhere(np.isin(result[:, 0], nIds, assume_unique=True)).ravel()

            return fndIds, result[fndIds, 1:]

    def getElementResult(self, increment: int,
                         resultKey: ResultsValue,
                         elIds: Optional[np.array] = None) -> Tuple[np.array, np.array, np.array]:
        """
        Returns a element result at step increment, for a correpsonding ResultsValue type.
        The elIDs parameter is optional and will select those values stored at these elements. The return value is
        a tuple, consisting of the element ids, integration points and corresponding result values.

        :param increment: The selected increment index available
        :param resultKey: A Valid Nodal Quantity in ResultsValue
        :param elIds: A list of element ids

        :return: A tuple of element ids, integration points and corresponding result values
        """
        if not self.hasResults():
            raise Exception('No results have been read')

        resultIncrement = self._increments.get(increment, None)

        if resultIncrement is None:
            raise Exception(f"Increment {increment} does not exist")

        if resultKey not in [ResultsValue.ELSTRESS, ResultsValue.ELHEATFLUX]:
            raise Exception('Invalid result key specified - not a Element Result ')

        result = resultIncrement.get(resultKey, None)

        if result is None:
            raise Exception('Result {:s} does not exist in increment {:d}'.format(resultKey, id))

        if elIds is None:
            # Return the results including the nodal ids
            return result[:, 0], result[:, 1], result[:, 2:]
        else:
            if isinstance(elIds, ElementSet):
                eIds = elIds.els  # Convert FEA ids to Pythonic indexing
            else:
                eIds = elIds

            # Perform a set intersection
            fndIds = np.argwhere(np.isin(result[:, 0], eIds, assume_unique=False)).ravel()

            return fndIds, result[fndIds, 1], result[fndIds, 2:]

    def hasResults(self) -> bool:
        return len(self.increments.keys()) > 0

    @property
    def numIncrements(self):
        """
        Convenience property for the number of increments available in the Calculix results file
        """
        return len(self.increments.keys())

    def lastIncrement(self):
        """
        Returns the last or final increment stored in the Calculix results file
        """

        idx = sorted(list(self._increments.keys()))[-1]
        return self._increments[idx]

    def findIncrementByTime(self, incTime, tol: Optional[float] = 1e-6) -> Tuple[int, Dict]:
        """
        Finds an increment at a stored time witin a specified tolerance (default: 1e-6)

        :param incTime: The specified analysis time to locate the increment
        :param tol: The numerical tolerance to find the increment [default: 1e-6]
        :return: The Increment data structure if found
        """
        for inc, increment in self._increments.items():
            if abs(increment['time'] - incTime) < tol:
                return inc, increment
        else:
            raise ValueError('Increment could not be found at time <{:.9f}>s'.format(incTime))

    @staticmethod
    def _getVals(fstr: str, line: str):
        """
        Returns a list of typed items based on an input format string. Credit for
        the processing of the .dat file is based from the PyCalculix project.
        https://github.com/spacether/pycalculix

        :param fstr: C format string, commas separate fields
        :param line: line string to parse
        :return:  List of typed items extracted from the line
        """

        res = []
        fstr = fstr.split(',')
        thestr = str(line)
        for item in fstr:
            if item[0] == "'":
                # strip off the char quaotes
                item = item[1:-1]
                # this is a string entry, grab the val out of the line
                ind = len(item)
                fwd = thestr[:ind]
                thestr = thestr[ind:]
                res.append(fwd)
            else:
                # format is: 1X, A66, 5E12.5, I12
                # 1X is number of spaces
                (mult, ctype) = (1, None)
                m_pat = re.compile(r'^\d+')  # find multiplier
                c_pat = re.compile(r'[XIEA]')  # find character
                if m_pat.findall(item) != []:
                    mult = int(m_pat.findall(item)[0])
                ctype = c_pat.findall(item)[0]
                if ctype == 'X':
                    # we are dealing with spaces, just reduce the line size
                    thestr = thestr[mult:]
                elif ctype == 'A':
                    # character string only, add it to results
                    fwd = thestr[:mult].strip()
                    thestr = thestr[mult:]
                    res.append(fwd)
                else:
                    # IE, split line into m pieces
                    w_pat = re.compile(r'[IE](\d+)')  # find the num after char
                    width = int(w_pat.findall(item)[0])
                    while mult > 0:
                        # only add items if we have enough line to look at
                        if width <= len(thestr):
                            substr = thestr[:width]
                            thestr = thestr[width:]
                            substr = substr.strip()  # remove space padding

                            if ctype == 'I':
                                substr = int(substr)
                            elif ctype == 'E':
                                substr = float(substr)
                            res.append(substr)
                        mult -= 1
        return res

    @staticmethod
    def __get_first_dataline(infile):
        """
        Reads infile until a line with data is found, then returns it
        A line that starts with ' -1' has data
        """
        while True:
            line = infile.readline()
            if line[:3] == ' -1':
                return line

    def readNodeDisp(self, line, rfstr) -> tuple:
        """
        Reads the nodal displacement values from the .frd file
        """
        nid, ux, uy, uz = self._getVals(rfstr, line)[1:]
        return nid, ux, uy, uz

    def readNodeForce(self, line, rfstr):
        """
        Reads the nodal force values from the .frd file
        """
        nid, f_x, f_y, f_z = self._getVals(rfstr, line)[1:]
        return nid, f_x, f_y, f_z

    def readNodeFlux(self, line, rfstr) -> tuple:
        """
        Reads the nodal heat flux values from the .frd file
        """
        nid, f_x, f_y, f_z = self._getVals(rfstr, line)[1:]
        return nid, f_x, f_y, f_z

    def readNodeTemp(self, line, rfstr) -> tuple:
        """
        Reads the nodal temp values from the .frd file
        """
        nid, temp = self._getVals(rfstr, line)[1:]
        return nid, temp

    def readNodeStress(self, line, rfstr) -> tuple:
        """
        Reads the nodal stress values from the .frd file
        """
        nid, sxx, syy, szz, sxy, syz, szx = self._getVals(rfstr, line)[1:]
        return nid, sxx, syy, szz, sxy, syz, szx

    def readNodeStrain(self, line, rfstr) -> tuple:
        """
        Reads the nodal strain values from the .frd file
        """
        nid, exx, eyy, ezz, exy, eyz, ezx = self._getVals(rfstr, line)[1:]
        return nid, exx, eyy, ezz, exy, eyz, ezx

    def readElFlux(self, line, rfstr, time):
        """Saves element integration point stresses"""
        elFlux = self._getVals(rfstr, line)[1:]

        elId, intp, qx, qy, qz = self._getVals(rfstr, line)

        return elId, intp, qx, qy, qz

    def readElStress(self, line, rfstr, time):
        """Saves element integration point stresses"""

        elId, intp, sxx, syy, szz, sxy, syz, szx = self._getVals(rfstr, line)

        return elId, intp, sxx, syy, szz, sxy, syz, szx

    def readElResultBlock(self, infile, line):
        """
        eturns an array of line, mode, rfstr, time
        """

        words = line.strip().split()
        # add time if not present
        time = float(words[-1])

        # set mode
        rfstr = "I10,2X,I2,6E14.2"

        mode = 'stress'
        infile.readline()
        line = infile.readline()
        return [line, mode, rfstr, time]

    def readNodalResultsBlock(self, infile):
        """
        Returns an array of line, mode, rfstr, time
        """

        line = infile.readline()
        fstr = "1X,' 100','C',6A1,E12.5,I12,20A1,I2,I5,10A1,I2"
        tmp = self._getVals(fstr, line)
        # [key, code, setname, value, numnod, text, ictype, numstp, analys, format_]
        time, format_ = tmp[3], tmp[9]

        # set results format to short, long or binary
        # only short and long are parsed so far
        if format_ == 0:
            rfstr = "1X,I2,I5,6E12.5"
        elif format_ == 1:
            rfstr = "1X,I2,I10,6E12.5"
        elif format_ == 2:
            # binary
            pass

        # set the time
        # self.__store_time(time)

        # get the name to determine if stress or displ
        line = infile.readline()
        fstr = "1X,I2,2X,8A1,2I5"
        # [key, name, ncomps, irtype]
        ar2 = self._getVals(fstr, line)
        name = ar2[1]
        iteration = tmp[7]
        line = self.__get_first_dataline(infile)

        return [line, name, rfstr, time, iteration]

    def read(self) -> None:
        """
        Opens up the results files and parses the results files to load all data within each increment
        """
        self.clearResults()

        logging.info('Reading the results files}')
        infile = open('{:s}.frd'.format(self.jobName), 'r')
        logging.info('Loading nodal results from file: {:s}.frd'.format(self.jobName))

        mode = None
        time = 0.0
        rfstr = ''

        while True:
            line = infile.readline()
            if not line:
                break

            if '2C' in line:
                # Read nodal coordinates
                reSearch = re.search('\s+2C\s+(\d+)', line)
                numNodes = int(reSearch.group(1))

                nodes = []
                for i in range(numNodes):
                    line = infile.readline()
                    nid, x, y, z = self._getVals("1X,I2,I10,6E12.5", line)[1:]
                    nodes.append([nid, x, y, z])

                self._nodes = np.array(nodes)

            if '3C' in line:
                # Read elemental coordinates
                reSearch = re.search('\s+3C\s+(\d+)', line)
                numElements = int(reSearch[1])

                elements = []
                line = infile.readline()

                elIds = []
                eId = None
                eType = None

                while True:

                    if line[:3] == ' -3':
                        if len(elIds) > 0:
                            elements.append([eId, eType] + elIds)
                        break
                    elif line[:3] == ' -1':
                        if len(elIds) > 0:
                            elements.append([eId, eType] + elIds)
                        eId, eType, eGrp, eMat = self._getVals("1X,' -1',5A1,4I5", line)[2:]
                        elIds = []
                    elif line[:3] == ' -2':
                        elIds += self._getVals("1X,'-2',20I10", line)[1:]
                    else:
                        raise Exception('Error parsing .frd file')

                    line = infile.readline()

                self._elements = elements

            # set the results mode
            if '1PSTEP' in line:
                # we are in a results block
                arr = self.readNodalResultsBlock(infile)
                line, mode, rfstr, time, inc = arr
                inc = int(inc)

                if inc not in self._increments.keys():
                    self._increments[inc] = {'time' : time,
                                             ResultsValue.DISP: [],
                                             ResultsValue.ELSTRESS: [],
                                             ResultsValue.STRESS: [],
                                             ResultsValue.STRAIN: [],
                                             ResultsValue.FORCE: [],
                                             ResultsValue.TEMP: []}

            # set mode to none if we hit the end of a resuls block
            if line[:3] == ' -3':
                mode = None

            if not mode:
                continue

            if mode == 'DISP':
                self._increments[inc][ResultsValue.DISP].append(self.readNodeDisp(line, rfstr))
            elif mode == 'STRESS':
                self._increments[inc][ResultsValue.STRESS].append(self.readNodeStress(line, rfstr))
            elif mode == 'TOSTRAIN':
                self._increments[inc][ResultsValue.STRAIN].append(self.readNodeStrain(line, rfstr))
            elif mode == 'FORC':
                self._increments[inc][ResultsValue.FORCE].append(self.readNodeForce(line, rfstr))
            elif mode == 'NDTEMP':
                self._increments[inc][ResultsValue.TEMP].append(self.readNodeTemp(line, rfstr))

        infile.close()

        """ 
        Read the element post-processing file
        """
        self.readDat()

        resultKeys = [ResultsValue.DISP, ResultsValue.STRESS, ResultsValue.STRAIN, ResultsValue.FORCE, ResultsValue.TEMP]

        # Process the nodal blocks
        for inc in self._increments.values():
            for key in resultKeys:
                inc[key] = self.orderNodes(np.array(inc[key]))

                if key == ResultsValue.STRESS:

                    if len(inc[key]) == 0:
                        # empty value
                        continue

                    # Generate the von-mises stress if the nodal stress property is available
                    nodeStress = inc[key]
                    sigma = nodeStress[:, 1:]
                    sigma_v = self.calculateVonMises(sigma)
                    sigma_v = np.hstack([nodeStress[:, 0].reshape(-1, 1), sigma_v.reshape(-1, 1)])
                    inc[ResultsValue.VMSTRESS] = sigma_v

        logging.info('Results Processor: The following times have been read:')
        timeIncrements = [val['time'] for val in self._increments.values()]
        logging.info(', '.join([str(time) for time in timeIncrements]))

        if len(timeIncrements) == 0:
            logging.warning('Results file contains no results')

    @staticmethod
    def calculateVonMises(sigma):

        numComps = sigma.shape[1]

        if numComps == 3:
            # 2D Stress Tensor (Planar Stress)
            sigma_v = np.sqrt(sigma[:, 0]**2 - sigma[:, 0] * sigma[:, 1] + sigma[:, 1]**2 + 3.*sigma[:, 2]**2)
        elif numComps == 6:
            # 3D Stress Tensor
            sigma_v = 0.5 * np.sqrt( (sigma[:, 0] - sigma[:, 1])**2 +
                                     (sigma[:, 1] - sigma[:, 2])**2 +
                                     (sigma[:, 2] - sigma[:,0])**2 +
                                  6.*(sigma[:, 3]**2 + sigma[:, 4]**2 + sigma[:, 5]**2))
        else:
            raise Exception('Invalid number of stress components')

        return sigma_v

    def clearResults(self):
        self._increments = {}
        self._elements = None
        self._nodes = None

    @staticmethod
    def orderNodes(nodeVals):
        if nodeVals.size == 0:
            return nodeVals

        return nodeVals[np.argsort(nodeVals[:, 0]), :]

    @staticmethod
    def orderElements(elVals):
        if elVals.size == 0:
            return elVals

        return elVals[np.argsort(elVals[:, 0]), :]

    def readDat(self) -> None:
        """
        Internal method that reads the Analysis' Calculix .dat file for the elemental quanties and parses the results.
        """
        fname = '{:s}.dat'.format(self.jobName)

        if not os.path.isfile(fname):
            raise Exception('Error: %s file not found' % fname)

        infile = open(fname, 'r')
        logging.info('Loading element results from file: {:s}'.format(fname))

        mode = None
        rfstr = ''
        incTime = 0.0
        inc = -1

        while True:
            line = infile.readline()

            if not line:
                break

            if line.strip() == '':
                mode = None

            # check for stress, we skip down to the line data when
            # we call __modearr_estrsresults
            if 'stress' in line:
                arr = self.readElResultBlock(infile, line)
                line, mode, rfstr, incTime = arr

                # Store the element stress results across all integration quadrature points
                inc, increment = self.findIncrementByTime(incTime)
                mode = 'elStress'

            elif 'heat flux' in line:
                arr = self.readElResultBlock(infile, line)
                line, mode, rfstr, incTime = arr

                # store the heatlufx results
                inc, increment = self.findIncrementByTime(incTime)

                mode = 'elHeatFlux'
                self._increments[inc][mode] = []

                if mode and inc > -1:
                    self._increments[inc][mode].append(self.readElFlux(line, rfstr, incTime))

            # set mode to none if we hit the end of a resuls block
            if line.isspace():
                mode = None

            if not mode:
                continue

            if mode == 'elStress':
                self._increments[inc]['elStress'].append(self.readElStress(line, rfstr, incTime))
            elif mode == 'elHeatFlux':
                self._increments[inc]['elHeatFlux'].append(self.readElFlux(line, rfstr, incTime))

        for inc in self._increments.values():

            if 'elStress' in inc:
                inc['elStress'] = self.orderElements(np.array(inc['elStress']))

            if 'elHeatFlux' in inc:
                inc['elHeatFlux'] = self.orderElements(np.array(inc['elHeatFlux']))

        infile.close()
