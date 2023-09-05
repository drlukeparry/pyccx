from enum import Enum, Flag, auto
from typing import Any, List, Tuple

import xml.etree.ElementTree as ET
import numpy as np

from ..core import ElementSet, NodeSet, SurfaceSet, DOF
from ..results import ResultProcessor

def exportToVTK(filename: str, results: ResultProcessor, inc: int = -1):
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
        2: [0,2,1,3,5,4],
        4: [0,1,2,3,4,5,6,7,8,9,10,11,16,17,18,19,12,13,14,15],
        5: [0,2,1,3,5,4]
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
    ePiece = ET.SubElement(e1, 'Piece', NumberOfPoints = str(len(results.nodes)),
                                        NumberOfCells = str(len(results.elements)))

    ePointData = ET.SubElement(ePiece, 'PointData')

    """ Write the Node Displacement Data """
    if len(resultIncrement['disp']) > 0:

        eDispArray = ET.SubElement(ePointData, 'DataArray', type="Float32", Name="Displacement", NumberOfComponents="3", Format="Ascii")
        nodeDispStr = ''

        for row in resultIncrement['disp']:
            nodeDispStr  += ' '.join([str(val) for val in row[1:]]) + '\n'
        eDispArray.text = nodeDispStr


    """ Write the Node RF Data """
    if len(resultIncrement['force']) > 0:
        eRFArray = ET.SubElement(ePointData, 'DataArray', type="Float32", Name="RF", NumberOfComponents="3", Format="Ascii")
        nodeDispStr = ''
        for row in resultIncrement['force']:
            nodeDispStr  += ' '.join([str(val) for val in row[1:]]) + '\n'
        eRFArray.text = nodeDispStr

    """ Write the Cauchy Stress Data """
    if len(resultIncrement['stress']) > 0:
        sigma = resultIncrement['stress'][:,1:]
        eSigmaArray = ET.SubElement(ePointData, 'DataArray', type="Float32", Name="stress",
                                    NumberOfComponents=str(sigma.shape[1]), Format="Ascii")
        nodeSigmaStr = ''
        for row in sigma:
            nodeSigmaStr  += ' '.join([str(val) for val in row]) + '\n'
        eSigmaArray.text = nodeSigmaStr

    """ Write strain data """
    if len(resultIncrement['strain']) > 0:
        eStrainArray = ET.SubElement(ePointData, 'DataArray', type="Float32", Name="strain", NumberOfComponents="6", Format="Ascii")
        nodeStrainStr = ''
        for row in resultIncrement['strain']:
            nodeStrainStr  += ' '.join([str(val) for val in row[1:]]) + '\n'
        eStrainArray.text = nodeStrainStr

    """ Export the remaining geometrical element information to the .vtu format"""
    eCellData = ET.SubElement(ePiece, 'CellData')

    ePoints = ET.SubElement(ePiece, 'Points')
    ePointsArray = ET.SubElement(ePoints, 'DataArray', type="Float32", Name="Points", NumberOfComponents="3", Format="Ascii")

    """ Write the Node Coordinate Data """
    nodeStr = ''
    for row in results.nodes:
        nodeStr  += ' '.join([str(val) for val in row[1:]]) + '\n'

    ePointsArray.text = nodeStr

    eCells = ET.SubElement(ePiece, 'Cells')
    eConArray  = ET.SubElement(eCells, 'DataArray', type="Int32", Name="connectivity", Format="Ascii")

    """
    Write the Node Coordinate Data:
    Note: Row is the element id, element type, element nodes
    """
    elConStr = ''
    for row in results.elements:
        # Note (row[1]) is the element type

        if row[1] in nodeMap:
            # Remap the nodes of the elements for special types in VTK
            elIds = np.array(row[2:])
            elIds = elIds[np.array(nodeMap[row[1]])]
            elConStr += ' '.join([str(val-1) for val in elIds]) + '\n'
        else:
            # Write connectivity directly
            elConStr += ' '.join([str(val-1) for val in row[2:]]) + '\n'

    eConArray.text = elConStr

    """ Write the element offset array """
    eOffArray  = ET.SubElement(eCells, 'DataArray', type="Int32", Name="offsets", Format="Ascii")
    elOffset = np.cumsum([len(row)-2 for row in results.elements])
    eOffArray.text = ' '.join([str(int(val)) for val in elOffset]) + '\n'

    """ Write the element type array """
    eTypeArray = ET.SubElement(eCells, 'DataArray', type="UInt8", Name="types", Format="Ascii")
    eTypes = [vtkMap[row[1]] for row in results.elements]
    eTypeArray.text = ' '.join([str(int(val)) for val in eTypes]) + '\n'

    """ Write the binary string to the fikle """
    b_xml = ET.tostring(data)
    with open(filename, 'wb') as f:
        f.write(b_xml)

    # Opening a file under the name
