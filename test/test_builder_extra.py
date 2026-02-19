import os
import tempfile
import numpy as np
from record_msg.builder import TransformBuilder, LocalizationBuilder, IMUBuilder, PointCloudBuilder


def test_transform_builder_basic():
    b = TransformBuilder()
    t = [0.1, 0.2, 0.3]
    r = [1.0, 0.0, 0.0, 0.0]
    pb = b.build('f', 'child', t, r, None)
    assert pb is not None
    assert pb.transforms[0].child_frame_id == 'child'


def test_localization_builder_basic():
    b = LocalizationBuilder()
    trans = [1.0, 2.0, 3.0]
    rot = [1.0, 0.0, 0.0, 0.0]
    pb = b.build(trans, rot, heading=0.5, t=123.0)
    assert pb is not None
    assert pb.pose.position.x == 1.0
    assert pb.measurement_time == 123.0


def test_imu_builder_sets_measurement_time():
    b = IMUBuilder()
    lin = [0.1, 0.2, 0.3]
    ang = [0.01, 0.02, 0.03]
    pb = b.build(lin, ang, t=1000.0)
    assert pb is not None
    assert hasattr(pb, 'measurement_time')


def test_pointcloud_builder_nuscenes_tmpfile(tmp_path):
    # Create a small binary file with two points (x,y,z,intensity)
    data = np.array([[1.0, 2.0, 3.0, 10.0], [4.0, 5.0, 6.0, 20.0]], dtype=np.float32)
    bin_path = tmp_path / 'scan.bin'
    data.tofile(str(bin_path))
    b = PointCloudBuilder(dim=4)
    pb = b.build_nuscenes(str(bin_path), frame_id='pcf')
    assert pb is not None
    assert pb.width == 2
    assert pb.point[0].x != 0
