import quaternion
import numpy as np
from atlasrl.motions.QuaternionToAtlasEuler import convertQuaternionToAtlasEuler

def test_identity():
    assert np.all((convertQuaternionToAtlasEuler(quaternion.one) - np.array((0, 0, 0))) < 1e-5)

def test_pos1():
    assert np.linalg.norm(convertQuaternionToAtlasEuler(quaternion.from_rotation_vector(np.array((0, np.pi / 2, 0)))) - np.array((0, 0, np.pi / 2))) < 1e-5

def test_pos2():
    assert np.linalg.norm(convertQuaternionToAtlasEuler(quaternion.from_rotation_vector(np.array((np.pi / 2, 0, 0)))) - np.array((np.pi / 2, 0, 0))) < 1e-5

def test_pos3():
    assert np.linalg.norm(convertQuaternionToAtlasEuler(quaternion.from_rotation_vector(np.array((0, 0, np.pi / 2)))) - np.array((0, -np.pi / 2, 0))) < 1e-5
