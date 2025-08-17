import sys
import math

import pytest
import pyccx
import numpy as np

class TestBasic:

    def test_version(self):

        assert pyccx.__version__