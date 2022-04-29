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


from record_msg.builder import ImageBuilder, PointCloudBuilder


def test_image_builder():
  img_path = 'test.jpg'
  image_builder = ImageBuilder()
  pb_image = image_builder.build(img_path, encoding='rgb8')
  print(pb_image)


def test_point_cloud_builder():
  point_cloud_path = 'test.pcd'
  point_cloud_builder = PointCloudBuilder()
  pb_point_cloud = point_cloud_builder.build(point_cloud_path)
  print(pb_point_cloud)


if __name__ == '__main__':
  # test_image_builder()
  test_point_cloud_builder()
