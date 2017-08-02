"""
This file tests the Molecule in the quest.
"""

import quest
import pytest
import psi4
import numpy as np


def test_molecule():
    testmol = quest.molecule.Molecule(quest.mollib["h2o"], "sto-3g")
    assert testmol.nel == 10

