import numpy as np
from record_msg import pypcd


def test_encode_decode_rgb_roundtrip():
    rgb = np.array([[10, 20, 30], [255, 128, 0]], dtype=np.uint8)
    packed = pypcd.encode_rgb_for_pcl(rgb)
    decoded = pypcd.decode_rgb_from_pcl(packed)
    assert decoded.shape == rgb.shape
    assert np.array_equal(decoded, rgb)


def test_make_xyz_point_cloud_and_buffer_roundtrip():
    xyz = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    pc = pypcd.make_xyz_point_cloud(xyz)
    # Basic sanity checks on returned PointCloud
    assert pc.width == 2
    assert pc.points == 2
    cp = pc.copy()
    assert cp.points == pc.points


def test_make_xyz_rgb_and_label_point_cloud_exceptions():
    # wrong dtype for xyz_rgb should raise
    xyz_rgb = np.zeros((2, 4), dtype=np.float64)
    try:
        pypcd.make_xyz_rgb_point_cloud(xyz_rgb)
        assert False, 'expected ValueError for non-float32 input'
    except ValueError:
        pass
    # label type invalid
    try:
        pypcd.make_xyz_label_point_cloud(np.zeros((2, 4)), label_type='z')
        assert False, 'expected ValueError for bad label_type'
    except ValueError:
        pass
