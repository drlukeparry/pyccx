import numpy as np
import abc

class Material(abc.ABC):
    """
    Base class for all material model definitions
    """
    MATERIALMODEL = 'INVALID'

    def __init__(self, name):
        self._input = ''
        self._name = name
        self._materialModel = ''

    @property
    def name(self) -> str:
        return self._name

    def setName(self, matName: str):
        self._name = matName

    @property
    @abc.abstractmethod
    def materialModel(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def writeInput(self):
        raise NotImplemented()

    @abc.abstractmethod
    def isValid(self) -> bool:
        """
        Abstract method - reimplement in material models to check parameters are correct by the user

        :return: bool
        """
        raise NotImplemented()


class ElasticMaterial(Material):
    """
    Represents a generic non-linear elastic material which may be used in both structural, and thermal type analyses
    """

    def __init__(self, name):

        super().__init__(name)

        self._E = 210e3
        self._nu = 0.33
        self._density = 7.85e-9
        self._alpha_CTE = 12e-6
        self.k = 50.0
        self._cp = 50.0

    @property
    def E(self):
        """Elastic Modulus :math:`E`"""
        return self._E

    @E.setter
    def E(self, val):
        self._E = val

    @property
    def nu(self):
        """Poisson's Ratio :math:`\\nu` """
        return self._nu

    @nu.setter
    def nu(self, val):
        self._nu = val

    @property
    def density(self):
        """Density :math:`\\rho`"""
        return self._density

    @density.setter
    def density(self, val):
        self._density = val

    @property
    def alpha_CTE(self):
        """Thermal Expansion Coefficient :math:`\\alpha_{cte}`"""
        return self._alpha_CTE

    @alpha_CTE.setter
    def alpha_CTE(self, val):
        self._alpha_CTE = val

    @property
    def k(self):
        """Thermal conductivity :math:`k`"""
        return self._k

    @k.setter
    def k(self, val):
        self._k = val

    @property
    def cp(self):
        """Specific Heat :math:`c_p`"""
        return self._cp

    @cp.setter
    def cp(self, val):
        self._cp = val

    @property
    def materialModel(self):
        return 'elastic' # Calculix material model

    def cast2Numpy(self, tempVals):
        if type(tempVals) == float:
            tempVal = np.array([tempVals])
        elif type(tempVals) == list:
            tempVal = np.array(tempVals)
        elif type(tempVals) == np.ndarray:
            tempVal = tempVals
        else:
            raise ValueError('Mat prop type not supported')

        return tempVal

    def writeElasticProp(self) -> str:

        lineStr = '*elastic'
        nu = self.cast2Numpy(self.nu)
        E = self.cast2Numpy(self.E)

        if (nu.ndim != E.ndim):
            raise ValueError("Both Poissons ratio and Young's modulus must be temperature dependent or constant")

        if nu.shape[0] == 1:
            if (nu.shape[0] != E.shape[0]):
                raise ValueError("Same number of entries must exist for Poissons ratio and Young' Modulus")

            lineStr += ',type=iso\n'
            if nu.ndim == 1:
                lineStr += '{:e},{:e}\n'.format(E[0], nu[0])
            elif nu.ndim == 2:
                for i in range(nu.shape[0]):
                    lineStr += '{:e},{:e},{:e}\n'.format(E[i, 1], nu[i, 1], E[0])
        else:
            raise ValueError('Not currently support elastic mode')

        return lineStr

    def writeMaterialProp(self, matPropName: str, tempVals) -> str:
        """
        Helper method to write the material property name and formatted values depending on the anisotropy of the material
        and if non-linear parameters are used.

        :param matPropName: str: Material property
        :param tempVals: Values to assign material properties
        :return: str:
        """

        if type(tempVals) == float:
            tempVal = np.array([tempVals])
        elif type(tempVals) == list:
            tempVal = np.array(tempVals)
        elif type(tempVals) == np.ndarray:
            tempVal = tempVals
        else:
            raise ValueError('Material prop type not supported')

        lineStr = '*{:s}'.format(matPropName)

        if (tempVal.ndim == 1 and tempVal.shape[0] == 1) or (tempVal.ndim == 2 and tempVal.shape[1] == 1):
            lineStr += ',type=iso\n'
        elif (tempVal.ndim == 1 and tempVal.shape[0] == 3) or (tempVal.ndim == 2 and tempVal.shape[1] == 4):
            lineStr += ',type=ortho\n'
        else:
            raise ValueError('Invalid mat property({:s}'.format(matPropName))

        if tempVal.ndim == 1:
            if tempVal.shape[0] == 1:
                lineStr += '{:e}\n'.format(tempVal[0])
            elif tempVal.shape[0] == 3:
                lineStr += '{:e},{:e},{:e}\n'.format(tempVal[0], tempVal[1], tempVal[2])

        if tempVal.ndim == 2:
            for i in range(tempVal.shape[0]):
                if tempVal.shape[1] == 2:
                    lineStr += '{:e},{:e}\n'.format(tempVal[i, 1], tempVal[i, 0])
                elif tempVal.shape[1] == 4:
                    lineStr += '{:e},{:e},{:e},{:e}\n'.format(tempVal[1], tempVal[2], tempVal[3], tempVal[0])

        return lineStr

    def isValid(self) -> bool:
        return True

    def writeInput(self) -> str:

        inputStr = '*material, name={:s}\n'.format(self._name)
        inputStr += '*{:s}\n'.format(self.materialModel)

        inputStr += self.writeElasticProp()

        if self.density:
            inputStr += self.writeMaterialProp('density', self.density)

        if self.cp:
            inputStr += self.writeMaterialProp('specific heat', self.cp)

        if self.alpha_CTE:
            inputStr += self.writeMaterialProp('expansion', self.alpha_CTE)

        if self.k:
            inputStr += self.writeMaterialProp('conductivity', self.k)

        return inputStr
