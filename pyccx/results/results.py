from typing import List, Tuple, Union

import abc
import re
import os
import logging
import numpy as np

from ..core import ElementSet, NodeSet


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
        raise NotImplemented()


class NodalResult(Result):

    def __init__(self, nodeSet: Union[str,NodeSet]):

        if not (isinstance(nodeSet, NodeSet) or isinstance(nodeSet, str)):
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
        """ Include temperature in the output """
        return self._displacement

    @displacement.setter
    def displacement(self, state: bool):
        self._displacement = state

    @property
    def temperature(self) -> bool:
        """ Include temperature in the output """
        return self._temperature

    @temperature.setter
    def temperature(self, state: bool):
        self._temperature = state

    @property
    def reactionForce(self) -> bool:
        """ Include reaction forces in the output """
        return self._reactionForce

    @reactionForce.setter
    def reactionForce(self, state: bool):
        self._reactionForce = state

    @property
    def heatFlux(self) -> bool:
        """ Include heat flux in the output """
        return self._heatFlux

    @heatFlux.setter
    def heatFlux(self, state):
        self._heatFlux = state

    @property
    def cauchyStress(self) -> bool:
        """ Include cauchy stress components in the output """
        return self._cauchyStress

    @cauchyStress.setter
    def cauchyStress(self, state):
        self._cauchyStress = state

    @property
    def strain(self) -> bool:
        """ Include strain in the output """
        return self._strain

    @strain.setter
    def strain(self, state):
        self._strain = state

    @property
    def plasticStrain(self) -> bool:
        """ Include plastic strain in the output """
        return self._plasticStrain

    @plasticStrain.setter
    def plasticStrain(self, state):
        self._plasticStrain = state

    @property
    def expandShellElements(self):
        """ Expand shell elements from their mid-surface in the output """
        return self._expandShellElements

    @expandShellElements.setter
    def expandShellElements(self, state):
        self._expandShellElements = state

    @property
    def nodeSet(self) -> NodeSet:
        """
        The node set to obtain values for post-processing.
        """
        return self._nodeSet

    @nodeSet.setter
    def nodeSet(self, nodeSet: NodeSet):

        if not (isinstance(nodeSet, NodeSet) or isinstance(nodeSet, str)):
            raise TypeError('NodalResult nodeset must be a NodeSet object or name of set')

        self._nodeSet = nodeSet

    def writeInput(self):
        inputStr = ''
        inputStr += '*NODE FILE, '

        if isinstance(self.nodeSet, str) and self.nodeSet != '':
            inputStr += 'NSET={:s}, '.format(self.nodeSet)

        if isinstance(self.nodeSet, NodeSet):
            inputStr += 'NSET={:s}, '.format(self.nodeSet.name)

        if not self._expandShellElements:
            inputStr += 'OUTPUT=2D, '

        inputStr += 'FREQUENCY={:d}\n'.format(self._frequency)

        if self._displacement:
            inputStr += 'U\n'

        if self._temperature:
            inputStr += 'NT\n'

        if self._reactionForce:
            inputStr += 'RF\n'

        inputStr += self.writeElementInput()

        return inputStr

    def writeElementInput(self):
        str = '*EL FILE, NSET={:s}, FREQUENCY={:d}\n'.format(self._nodeSet.name, self._frequency)

        if self._cauchyStress:
            str += 'S\n'

        if self._strain:
            str += 'E\n'

        if self._plasticStrain:
            str += 'PEEQ\n'

        if self._heatFlux:
            str += 'HFL\n'

        return str


class ElementResult(Result):
    """

    Including this object will inform Calcuix to save the elemental integration properties to the file (.dat) for
    the selected ElementSet.

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
        """ The equivalent plastic strain"""
        return self._plasticStrain

    @plasticStrain.setter
    def plasticStrain(self, state: bool):
        self._plasticStrain = state

    @property
    def strain(self) -> bool:
        """
        The total Lagrangian strain for (hyper)elastic materials and incremental plasticity
        and the total Eulerian strain for deformation plasticity."""
        return self._strain

    @strain.setter
    def strain(self, state: bool ):
        self._strain = state

    @property
    def mechanicalStrain(self) -> bool:
        """
        This is the mechanical Lagrangian strain for (hyper)elastic materials and incremental plasticity and the
        mechanical Eulerian strain for deformation plasticity (mechanical strain = total strain - thermal strain)."""
        return self._mechanicalStrain

    @mechanicalStrain.setter
    def mechanicalStrain(self, state: bool):
        self._mechanicalStrain = state

    @property
    def ESE(self) -> bool:
        """ """
        return self._ESE

    @ESE.setter
    def ESE(self, state: bool):
        """ The internal energy per unit volume """
        self._ESE = state

    @property
    def heatFlux(self) -> bool:
        """ Include heat flux in the output """
        return self._heatFlux

    @heatFlux.setter
    def heatFlux(self, state):
        self._heatFlux = state

    @property
    def cauchyStress(self) -> bool:
        """ Include cauchy stress components in the output """
        return self._cauchyStress

    @cauchyStress.setter
    def cauchyStress(self, state):
        self._cauchyStress = state

    @property
    def elementSet(self) -> ElementSet:
        """
        The elementset to obtain values for post-processing.
        """

        return self._elSet

    @elementSet.setter
    def elementSet(self, elSet: ElementSet):

        if not (isinstance(elSet, NodeSet) or isinstance(elSet, str)):
            raise TypeError('ElementResult must be initialized with a NodeSet object or name of set')

        self._elSet = elSet

    def writeInput(self):

        str = ''

        elName = self._elSet.name if isinstance(self._elSet, ElementSet) else self._elSet

        str += '*EL PRINT, ELSET={:s}, FREQUENCY={:d}\n'.format(elName, self._frequency)

        if self._cauchyStress:
            str += 'S\n'

        if self._strain:
            str += 'E\n'

        if self._mechanicalStrain:
            str += 'ME\n'

        if self._plasticStrain:
            str += 'PEEQ\n'

        if self._ESE:
            str += 'ELSE\n'

        if self._heatFlux:
            str += 'HFL\n'

        return str


class ResultProcessor:
    """
    ResultProcessor takes the output (results) file from the Calculix simulation and processes the ASCII .frd file
    to load the results into a structure. Individual timesteps (increments) are segregated and may be accessed
    accordingly.
    """

    def __init__(self, jobName):

        self.increments = {}
        self.jobName = jobName

        self._elements = None
        self._nodes = None

        logging.info('Reading file {:s}'.format(jobName))

    @property
    def nodes(self):
        """
        Nodes identified in the Calculix results file
        """
        return self._nodes

    @property
    def elements(self):
        """
        Elements identified in the Calculix results file
        """
        return self._elements

    def lastIncrement(self):
        """
        Returns the last increment of the Calculix results file

        :return: The last increment of the Calculix results file
        """

        idx = sorted(list(self.increments.keys()))[-1]
        return self.increments[idx]

    def findIncrementByTime(self, incTime) -> int:

        for inc, increment in self.increments.items():
            if abs(increment['time'] - incTime) < 1e-9:
                return inc, increment
        else:
            raise ValueError('Increment could not be found at time <{:.5f}>s'.format(incTime))

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
        nid, ux, uy, uz = self._getVals(rfstr, line)[1:]
        return nid, ux, uy, uz

    def readNodeForce(self, line, rfstr):
        nid, f_x, f_y, f_z = self._getVals(rfstr, line)[1:]
        return nid, f_x, f_y, f_z

    def readNodeFlux(self, line, rfstr) -> tuple:
        nid, f_x, f_y, f_z = self._getVals(rfstr, line)[1:]
        return nid, f_x, f_y, f_z

    def readNodeTemp(self, line, rfstr) -> tuple:
        nid, temp = self._getVals(rfstr, line)[1:]
        return nid, temp

    def readNodeStress(self, line, rfstr) -> tuple:
        nid, sxx, syy, szz, sxy, syz, szx = self._getVals(rfstr, line)[1:]
        return nid, sxx, syy, szz, sxy, syz, szx

    def readNodeStrain(self, line, rfstr) -> tuple:
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
        """Returns an array of line, mode, rfstr, time"""

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
        Opens up the results files and processes the results
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

                if inc not in self.increments.keys():
                    self.increments[inc] = {'time'  : time,
                                            'disp'  : [],
                                            'elStress': [],
                                            'stress': [],
                                            'strain': [],
                                            'force' : [],
                                            'temp'  : []}

            # set mode to none if we hit the end of a resuls block
            if line[:3] == ' -3':
                mode = None

            if not mode:
                continue

            if mode == 'DISP':
                self.increments[inc]['disp'].append(self.readNodeDisp(line, rfstr))
            elif mode == 'STRESS':
                self.increments[inc]['stress'].append(self.readNodeStress(line, rfstr))
            elif mode == 'TOSTRAIN':
                self.increments[inc]['strain'].append(self.readNodeStrain(line, rfstr))
            elif mode == 'FORC':
                self.increments[inc]['force'].append(self.readNodeForce(line, rfstr))
            elif mode == 'NDTEMP':
                self.increments[inc]['temp'].append(self.readNodeTemp(line, rfstr))

        infile.close()

        """ 
        Read the element post-processing file
        """
        self.readDat()

        # Process the nodal blocks
        for inc in self.increments.values():
            inc['disp'] = self.orderNodes(np.array(inc['disp']))
            inc['stress'] = self.orderNodes(np.array(inc['stress']))
            inc['strain'] = self.orderNodes(np.array(inc['strain']))
            inc['force'] = self.orderNodes(np.array(inc['force']))
            inc['temp'] = self.orderNodes(np.array(inc['temp']))

        logging.info('Results Processor: The following times have been read:')
        timeIncrements = [val['time'] for val in self.increments.values()]
        logging.info(', '.join([str(time) for time in timeIncrements]))

    def clearResults(self):
        self.increments = {}
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

    def readDat(self):

        fname = '{:s}.dat'.format(self.jobName)

        if not os.path.isfile(fname):
            raise Exception('Error: %s file not found' % fname)
            return

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
                self.increments[inc][mode] = []

                if mode and inc > -1:
                    self.increments[inc][mode].append(self.readElFlux(line, rfstr, incTime))

            # set mode to none if we hit the end of a resuls block
            if line.isspace():
                mode = None

            if not mode:
                continue

            if mode == 'elStress':
                self.increments[inc]['elStress'].append(self.readElStress(line, rfstr, incTime))
            elif mode == 'elHeatFlux':
                self.increments[inc]['elHeatFlux'].append(self.readElFlux(line, rfstr, incTime))

        for inc in self.increments.values():

            if 'elStress' in inc:
                inc['elStress'] = self.orderElements(np.array(inc['elStress']))

            if 'elHeatFlux' in inc:
                inc['elHeatFlux'] = self.orderElements(np.array(inc['elHeatFlux']))

        infile.close()