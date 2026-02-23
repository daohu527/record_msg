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
import types
import pytest


@pytest.fixture
def small_rgb_image(tmp_path):
    # create a 2x2 RGB image array
    arr = np.zeros((2, 2, 3), dtype=np.uint8)
    arr[:] = [10, 20, 30]
    return arr


@pytest.fixture
def fake_cv2_module(small_rgb_image):
    m = types.SimpleNamespace()

    def imread(fname, flag):
        # ignore fname/flag and return the synthetic image
        return small_rgb_image

    # minimal flags used by ImageBuilder
    m.IMREAD_COLOR = 1
    m.IMREAD_GRAYSCALE = 0
    m.imread = imread
    return m


class FakePointCloud:
    def __init__(self, points):
        self.width = len(points)
        self.height = 1
        # store as iterable of tuples (x,y,z,intensity,timestamp)
        self.pc_data = points


@pytest.fixture
def fake_pypcd_module():
    m = types.SimpleNamespace()

    def point_cloud_from_path(fname):
        # return a tiny pointcloud with two points and timestamps
        pts = [(1.0, 2.0, 3.0, 10, 1.234), (4.0, 5.0, 6.0, 20, 2.345)]
        return FakePointCloud(pts)

    m.point_cloud_from_path = point_cloud_from_path
    return m


@pytest.fixture(autouse=True)
def patch_builder_modules(
    monkeypatch, request, fake_cv2_module, fake_pypcd_module, small_rgb_image
):
    # Allow integration tests to opt-out of the autouse patching by adding
    # the marker @pytest.mark.integration to the test. Unit tests will be
    # patched to avoid disk I/O and cv2 dependency.
    if request.node.get_closest_marker("integration"):
        # integration tests should use real files and real modules
        yield
        return

    # Patch the modules used by record_msg.builder to avoid disk I/O and cv2 dependency
    import record_msg.builder as builder

    # set cv2 attribute if present or not (raising=False so attribute is created)
    monkeypatch.setattr(builder, "cv2", fake_cv2_module, raising=False)
    # patch pypcd used by builder
    monkeypatch.setattr(builder, "pypcd", fake_pypcd_module)
    # Patch PIL.Image.open used by builder to return an image constructed from the
    # synthetic numpy array so tests don't need real files.
    from PIL import Image as PILImage

    def _fake_open(fp):
        return PILImage.fromarray(small_rgb_image)

    monkeypatch.setattr(PILImage, "open", _fake_open)
    yield
