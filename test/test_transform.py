from record_msg import pypcd
import time
import numpy as np

def transform_mat(pc_data, T):
    return pc_data @ T

def transform_points(pc_data, T):
    results = []
    for point in pc_data:
        results.append(point @ T)
    return results

def test_transform():
    point_cloud_path = 'test.pcd'
    point_cloud = pypcd.point_cloud_from_path(point_cloud_path)
    pc_data = point_cloud.pc_data
    pc = np.column_stack([pc_data['x'], pc_data['y'], pc_data['z'], pc_data['intensity']])
    T_id = np.identity(4)
    T_rand = np.random.randn(4, 4)

    # check correctness
    assert np.allclose(transform_mat(pc, T_id), np.stack(transform_points(pc, T_id)))
    assert np.allclose(transform_mat(pc, T_rand), np.stack(transform_points(pc, T_rand)))

def compare_performance():
    T_id = np.identity(4)
    T_rand = np.random.randn(4, 4)

    pcs = np.random.randn(100, 10000, 4)
    
    # identity transform
    # by mat
    start = time.time()
    for pc in pcs:
        transform_mat(pc, T_id)
    end = time.time()
    t1 = end - start

    # by point
    start = time.time()
    for pc in pcs:
        transform_points(pc, T_id)
    end = time.time()
    t2 = end - start

    print(f'identity transform by points is {t2 / t1}x slower')

    # identity transform
    # by mat
    start = time.time()
    for pc in pcs:
        transform_mat(pc, T_rand)
    end = time.time()
    t1 = end - start

    # by point
    start = time.time()
    for pc in pcs:
        transform_points(pc, T_rand)
    end = time.time()
    t2 = end - start

    print(f'random transform by points is {t2 / t1}x slower')

if __name__ == '__main__':
    test_transform()
    compare_performance()
    