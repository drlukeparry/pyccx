
import os
import shutil
from typing import Optional
import xml.etree.ElementTree as ET
import numpy as np

from ..results import ResultProcessor, ResultsValue

def exportToPVD(filename: str, results: ResultProcessor):
    """
    Exports all the timestep increments to a pvd file for visualisation in Paraview

    :param filename: The root filename of the .pvd file
    :param results: The PyCCX results processor
    """

    rootFilename = os.path.basename(filename).split('.')[0]
    rootDir = os.path.dirname(filename)

    if rootDir:
        rootDir = rootDir + '/'

    data = ET.Element('VTKFile', type="Collection", version="0.1", byte_order="LittleEndian")
    colEl = ET.SubElement(data, 'Collection')

    resultsFolder = '{:s}{:s}-data'.format(rootDir, rootFilename)

    """ Remove the previous directory """
    try:
        shutil.rmtree(resultsFolder)
    except:
        pass

    os.mkdir(resultsFolder)

    colItems = []
    for inc in results.increments:

        # iterData = results.increments[inc]

        incPath = '{:s}/{:s}'.format(resultsFolder, 'increment-{:d}.vtu'.format(inc))

        """ Export the .vtu format to the data folder """
        exportToVTK(incPath, results, inc)

        dataSetEl = ET.SubElement(colEl, 'DataSet', timestep="{:d}".format(inc), group="", part="0", file=incPath)
        colItems.append(dataSetEl)

    b_xml = ET.tostring(data)
    with open('{:s}'.format(filename), 'wb') as f:
        f.write(b_xml)


def exportToVTK(filename: str, results: ResultProcessor, inc: Optional[int] = -1):
    """
    Exports a single time step result to the .vtu file in its xml format

    :param filename: filename to export to
    :param results: results object
    :param inc: selected time increment key to export
    """
    vtkMap = {
        1: 12,  # 8 node brick
        2: 13,  # 6 node wedge
        3: 10,  # 4 node tet
        4: 25,  # 20 node brick
        5: 13,  # 15 node wedge
        6: 24,  # 10 node tet
        7: 5,  # 3 node shell
        8: 22,  # 6 node shell
        9: 9,  # 4 node shell
        10: 23,  # 8 node shell
        11: 3,  # 2 node beam
        12: 21,  # 3 node beam
    }

    """
    Note: The node map is used to remap the nodes of the elements for special types in VTK
    This is for wedges and hex elements which have a different node ordering in VTK
    """
    nodeMap = {
        2: [0, 2, 1, 3, 5, 4],
        4: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19, 12, 13, 14, 15],
        5: [0, 2, 1, 3, 5, 4]
    }

    """ Select the result increment to export """
    if inc == -1:
        """ Last increment """
        resultIncrement = results.lastIncrement()
    else:
        """ Selected increment """
        if results.increments.get(inc, None) is None:
            raise ValueError('Selected increment ({:d}) does not exist in the results'.format(inc))

        resultIncrement = results.increments[inc]

    data = ET.Element('VTKFile', type="UnstructuredGrid")
    e1 = ET.SubElement(data, 'UnstructuredGrid')
    ePiece = ET.SubElement(e1, 'Piece', NumberOfPoints=str(len(results.nodes[0])),
                                                                 NumberOfCells=str(len(results.elements[0])))

    ePointData = ET.SubElement(ePiece, 'PointData')

    """ Write the Node Displacement Data """
    if len(resultIncrement[ResultsValue.DISP]) > 0:

        eDispArray = ET.SubElement(ePointData, 'DataArray', type="Float32",
                                                         Name="Displacement",
                                                         NumberOfComponents="3", Format="Ascii")
        nodeDispStr = ''

        for row in resultIncrement[ResultsValue.DISP]:
            nodeDispStr += ' '.join([str(val) for val in row[1:]]) + '\n'
        eDispArray.text = nodeDispStr

    """ Write the Node RF Data """
    if len(resultIncrement[ResultsValue.FORCE]) > 0:
        eRFArray = ET.SubElement(ePointData, 'DataArray', type="Float32",
                                                      Name="RF", NumberOfComponents="3", Format="Ascii")
        nodeDispStr = ''
        for row in resultIncrement[ResultsValue.FORCE]:
            nodeDispStr += ' '.join([str(val) for val in row[1:]]) + '\n'
        eRFArray.text = nodeDispStr

    """ Write the Cauchy Stress Data """
    if len(resultIncrement[ResultsValue.STRESS]) > 0:
        sigma = resultIncrement[ResultsValue.STRESS][:, 1:]
        eSigmaArray = ET.SubElement(ePointData, 'DataArray', type="Float32", Name="stress",
                                    NumberOfComponents=str(sigma.shape[1]), Format="Ascii")
        nodeSigmaStr = ''
        for row in sigma:
            nodeSigmaStr += ' '.join([str(val) for val in row]) + '\n'
        eSigmaArray.text = nodeSigmaStr

    """ Write the Cauchy Stress Data """
    if resultIncrement.get(ResultsValue.VMSTRESS, None) is not None:
        if len(resultIncrement[ResultsValue.VMSTRESS]) > 0:
            sigma = resultIncrement[ResultsValue.VMSTRESS][:, 1:]
            eSigmaVMArray = ET.SubElement(ePointData, 'DataArray', type="Float32",
                                                                      Name="stressVM", NumberOfComponents=str(sigma.shape[1]),
                                                                      Format="Ascii")
            nodeSigmaStr = ''
            for row in sigma:
                nodeSigmaStr += ' '.join([str(val) for val in row]) + '\n'

            eSigmaVMArray.text = nodeSigmaStr

    """ Write strain data """
    if len(resultIncrement[ResultsValue.STRAIN]) > 0:
        eStrainArray = ET.SubElement(ePointData, 'DataArray', type="Float32",
                                                            Name="strain", NumberOfComponents="6", Format="Ascii")
        nodeStrainStr = ''
        for row in resultIncrement[ResultsValue.STRAIN]:
            nodeStrainStr += ' '.join([str(val) for val in row[1:]]) + '\n'
        eStrainArray.text = nodeStrainStr

    """ Export the remaining geometrical element information to the .vtu format"""
    # eCellData = ET.SubElement(ePiece, 'CellData')

    ePoints = ET.SubElement(ePiece, 'Points')
    ePointsArray = ET.SubElement(ePoints, 'DataArray', type="Float32",
                                                        Name="Points", NumberOfComponents="3", Format="Ascii")

    """ Write the Node Coordinate Data """
    nodeStr = ''
    for row in results.nodes[1]:
        nodeStr += ' '.join([str(val) for val in row]) + '\n'

    ePointsArray.text = nodeStr

    eCells = ET.SubElement(ePiece, 'Cells')
    eConArray = ET.SubElement(eCells, 'DataArray', type="Int32", Name="connectivity", Format="Ascii")

    """
    Write the Node Coordinate Data:
    Note: Row is the element id, element type, element nodes
    """
    elConStr = ''

    elIds, elType, elCon = results.elements
    for i in range(len(elIds)):
        # Note (row[1]) is the element type

        if elType[i] in nodeMap:
            # Remap the nodes of the elements for special types in VTK
            elConIds = np.array(elCon[i])
            elConIds = elConIds[np.array(nodeMap[elType[i]])]
            elConStr += ' '.join([str(val-1) for val in elConIds]) + '\n'
        else:
            # Write connectivity directly
            elConStr += ' '.join([str(val-1) for val in elCon[i]]) + '\n'

    eConArray.text = elConStr

    """ Write the element offset array """
    eOffArray = ET.SubElement(eCells, 'DataArray', type="Int32", Name="offsets", Format="Ascii")
    elOffset = np.cumsum([len(row) for row in elCon])
    eOffArray.text = ' '.join([str(int(val)) for val in elOffset]) + '\n'

    """ Write the element type array """
    eTypeArray = ET.SubElement(eCells, 'DataArray', type="UInt8", Name="types", Format="Ascii")
    eTypes = [vtkMap[row] for row in elType]
    eTypeArray.text = ' '.join([str(int(val)) for val in eTypes]) + '\n'

    """ Write the binary string to the file """
    b_xml = ET.tostring(data)
    with open(filename, 'wb') as f:
        f.write(b_xml)

    # Opening a file under the name
