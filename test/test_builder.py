#!/usr/bin/env python

# Copyright 2022 daohu527 <daohu527@gmail.com>
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


import os
from record_msg.builder import ImageBuilder, PointCloudBuilder


def test_image_builder_creates_message(tmp_path):
    img_file = tmp_path / "img.jpg"
    # file path is not used by the patched imread, but builder expects a path
    img_file.write_text("")
    builder = ImageBuilder()
    pb_image = builder.build(str(img_file), frame_id="f0", encoding="rgb8")
    assert pb_image is not None
    assert pb_image.encoding == "rgb8"
    assert pb_image.width == 2 and pb_image.height == 2


def test_point_cloud_builder_creates_message(tmp_path):
    pcd_file = tmp_path / "test.pcd"
    pcd_file.write_text("")
    builder = PointCloudBuilder()
    pb_pc = builder.build(str(pcd_file), frame_id="pc0")
    assert pb_pc is not None
    assert pb_pc.width == 2
    assert pb_pc.height == 1
