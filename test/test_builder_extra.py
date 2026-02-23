#!/usr/bin/env python

# Copyright 2026 daohu527 <daohu527@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import pytest
from record_msg.builder import (
    TransformBuilder,
    LocalizationBuilder,
    IMUBuilder,
    PointCloudBuilder,
)


def test_transform_builder_basic():
    b = TransformBuilder()
    t = [0.1, 0.2, 0.3]
    r = [1.0, 0.0, 0.0, 0.0]
    pb = b.build("f", "child", t, r, None)
    assert pb is not None
    assert pb.transforms[0].child_frame_id == "child"


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
    assert hasattr(pb, "measurement_time")


def test_pointcloud_builder_nuscenes_tmpfile(tmp_path):
    # Create a small binary file with two points (x,y,z,intensity)
    data = np.array([[1.0, 2.0, 3.0, 10.0], [4.0, 5.0, 6.0, 20.0]], dtype=np.float32)
    bin_path = tmp_path / "scan.bin"
    data.tofile(str(bin_path))
    b = PointCloudBuilder(dim=4)
    pb = b.build_nuscenes(str(bin_path), frame_id="pcf")
    assert pb is not None
    assert pb.width == 2
    assert pb.point[0].x != 0


def test_pointcloud_builder_nuscenes_unsupported_extension_raises(tmp_path):
    invalid_path = tmp_path / "scan.invalid"
    invalid_path.write_text("x")
    b = PointCloudBuilder(dim=4)
    with pytest.raises(ValueError, match="Unsupported file extension"):
        b.build_nuscenes(str(invalid_path), frame_id="pcf")


def test_transform_builder_header_sets_zero_timestamp():
    b = TransformBuilder()
    pb = b.build("f", "child", [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], 0)
    header = pb.transforms[0].header
    assert header.timestamp_sec == 0
    if hasattr(header, "HasField"):
        assert header.HasField("timestamp_sec")
