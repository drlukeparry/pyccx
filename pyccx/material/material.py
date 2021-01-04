import numpy as np
import abc
from enum import Enum, auto


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
        Abstract method: re-implement in material models to check parameters are correct by the user
        """
        raise NotImplemented()


class ElastoPlasticMaterial(Material):
    """
    Represents a generic non-linear elastic/plastic material which may be used in both structural, and thermal type analyses
    """

    class WorkHardeningType(Enum):
        """
        Work hardening mode selecting the hardening regime for the accumulation of plastic-strain
        """

        NONE = auto()
        """ Prevents any plastic deformation """

        ISOTROPIC = auto()
        """ Isotropic  work hardening """

        KINEMATIC = auto()
        """ Kinematic work hardening """

        COMBINED = auto()
        """ Cyclic work hardening """

    def __init__(self, name):

        super().__init__(name)

        self._E = 210e3
        self._nu = 0.33
        self._density = 7.85e-9
        self._alpha_CTE = 12e-6
        self._k = 50.0
        self._cp = 50.0

        # Plastic Behavior
        self._workHardeningMode = ElastoPlasticMaterial.WorkHardeningType.NONE
        self._hardeningCurve = None

    @property
    def E(self):
        """Elastic Modulus :math:`E`

        The Young's Modulus :math:`E` can be both isotropic by setting as a scalar value, or orthotropic by
        setting to an (1x3) array corresponding to :math:`E_{ii}, E_{jj}, E_{kk}` for each direction. Temperature dependent
        Young's Modulus can be set by providing an nx4 array, where the 1st column is the temperature :math:`T`
        and the remaining columns are the orthotropic values of :math:`E`.
        """
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
        """Thermal Expansion Coefficient :math:`\\alpha_{cte}`

        The thermal conductivity :math:`alpha_{cte}` can be both isotropic by setting as a scalar value, or orthotropic by
        setting to an (1x3) array corresponding to :math:`\alpha_{cte}` for each direction. Temperature dependent thermal
        expansion coefficient can be set by providing an nx4 array, where the 1st column is the temperature :math:`T`
        and the remaining columns are the orthotropic values of :math:`\alpha_{cte}`.
        """
        return self._alpha_CTE

    @alpha_CTE.setter
    def alpha_CTE(self, val):
        self._alpha_CTE = val

    @property
    def k(self):
        """Thermal conductivity :math:`k`

        The thermal conductivity :math:`k` can be both isotropic by setting as a scalar value, or orthotropic by setting to an (1x3) array corresponding
        to :math:`k_{ii}, k_{jj}, k_{kk}` for each direction. Temperature dependent thermal conductivity eat can be set
        by providing an nx4 array, where the 1st column is the temperature :math:`T` and the remaining columns are the
        orthotropic values of :math:`k`.
        """
        return self._k

    @k.setter
    def k(self, val):
        self._k = val

    @property
    def cp(self):
        """Specific Heat :math:`c_p`

        The specific heat :math:`c_p` can be both isotropic by setting as a scalar value, or orthotropic by setting to an (1x3) array corresponding
        to :math:`c_p` for each direction. Temperature dependent specific heat can be set by providing an nx4 array,
        where the 1st column is the temperature :math:`T` and the remaining columns are the orthotropic values of :math:`c_p`.
        """
        return self._cp

    @cp.setter
    def cp(self, val):
        self._cp = val

    def isPlastic(self) -> bool:
        """
        Returns True if the material exhibits a plastic behaviour
        """
        return self._workHardeningMode is not ElastoPlasticMaterial.WorkHardeningType.NONE

    @property
    def workHardeningMode(self):
        """
        The work hardening mode of the material - if this is set, plastic behaviour will be assumed requiring a
        work hardening curve to be provided
        """
        return self._workHardeningMode

    @workHardeningMode.setter
    def workHardeningMode(self, mode: WorkHardeningType) -> None:
        self._workHardeningMode = mode

    @property
    def hardeningCurve(self) -> np.ndarray:
        """
        Sets the work hardening stress-strain curve with an nx3 array (curve) set with each row entry to
        (stress :math:`\\sigma`, plastic strain :math:`\\varepsilon_p`, Temperature :math:`T`. The first row
        of a temperature group describes the yield point :math:`\\sigma_y` for the onset of the plastic regime.
        """
        return self._hardeningCurve

    @hardeningCurve.setter
    def hardeningCurve(self, curve):
        if not isinstance(curve, np.ndarray) or curve.shape[1] != 3:
            raise ValueError('Work hardening curve should be an nx3 numpy array')

        self._hardeningCurve = curve

    @property
    def materialModel(self):
        return 'elastic' # Calculix material model

    @staticmethod
    def cast2Numpy(tempVals):
        if type(tempVals) == float:
            tempVal = np.array([tempVals])
        elif type(tempVals) == list:
            tempVal = np.array(tempVals)
        elif type(tempVals) == np.ndarray:
            tempVal = tempVals
        else:
            raise ValueError('Mat prop type not supported')

        return tempVal

    def _writeElasticProp(self) -> str:

        lineStr = '*elastic'
        nu = self.cast2Numpy(self.nu)
        E = self.cast2Numpy(self.E)

        if nu.ndim != E.ndim:
            raise ValueError("Both Poisson's ratio and Young's modulus must be temperature dependent or constant")

        if nu.shape[0] == 1:
            if nu.shape[0] != E.shape[0]:
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

    def _writePlasticProp(self):

        if not self.isPlastic():
            return ''

        if self.isPlastic() and self.hardeningCurve is None:
            raise ValueError('Plasticity requires a work hardening curve to be defined')

        lineStr = ''
        if self._workHardeningMode is ElastoPlasticMaterial.WorkHardeningType.ISOTROPIC:
            lineStr += '*plastic HARDENING=ISOTROPIC\n'
        elif self._workHardeningMode is ElastoPlasticMaterial.WorkHardeningType.KINEMATIC:
            lineStr += '*plastic HARDENING=KINEMATIC\n'
        elif self._workHardeningMode is ElastoPlasticMaterial.WorkHardeningType.COMBINED:
            lineStr += '*cyclic hardening HARDENING=COMBINED\n'

        for i in range(self.hardeningCurve.shape[0]):
            lineStr += '{:e},{:e},{:e}\n'.format(self._hardeningCurve[i, 0], # Stress
                                                 self._hardeningCurve[i, 1], # Plastic Strain
                                                 self._hardeningCurve[i, 2]) # Temperature

    def _writeMaterialProp(self, matPropName: str, tempVals) -> str:
        """
        Helper method to write the material property name and formatted values depending on the anisotropy of the material
        and if non-linear parameters are used.

        :param matPropName: Material property
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
            lineStr +=  '\n' #',type=iso\n'
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

        inputStr += self._writeElasticProp()

        if self._density:
            inputStr += self._writeMaterialProp('density', self._density)

        if self._cp:
            inputStr += self._writeMaterialProp('specific heat', self._cp)

        if self._alpha_CTE:
            inputStr += self._writeMaterialProp('expansion', self._alpha_CTE)

        if self._k:
            inputStr += self._writeMaterialProp('conductivity', self._k)

        # Write the plastic mode
        inputStr += self._writePlasticProp()

        return inputStr
