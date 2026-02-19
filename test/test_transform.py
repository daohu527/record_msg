import numpy as np


def transform_mat(pc_data, T):
    return pc_data @ T


def transform_points(pc_data, T):
    return np.vstack([point @ T for point in pc_data])


def test_transform_equivalence():
    np.random.seed(0)
    # small synthetic point set (n,4)
    pc = np.random.randn(10, 4)
    T_id = np.eye(4)
    T_rand = np.random.randn(4, 4)

    assert np.allclose(transform_mat(pc, T_id), transform_points(pc, T_id))
    assert np.allclose(transform_mat(pc, T_rand), transform_points(pc, T_rand))


def test_transform_handles_empty():
    pc = np.zeros((0, 4))
    T = np.eye(4)
    assert transform_mat(pc, T).shape[0] == 0
