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


import cv2
import time

from record_msg import pypcd

from modules.common.proto import header_pb2
from modules.drivers.proto import sensor_image_pb2, pointcloud_pb2


class Builder(object):
  def __init__(self) -> None:
    self._frame_id = 0

  def _build_header(self, header, t):
    header.timestamp_sec = t
    header.module_name = 'camera'
    header.sequence_num = self._frame_id
    header.camera_timestamp = int(t * 1e9)
    header.lidar_timestamp = int(t * 1e9)
    header.version = 1
    header.frame_id = str(self._frame_id)


class ImageBuilder(Builder):
  def __init__(self) -> None:
    super().__init__()

  def _to_flag(self, encoding):
    if encoding == 'rgb8' or encoding == 'bgr8':
      return cv2.IMREAD_COLOR
    elif encoding == 'gray' or encoding == 'y':
      return cv2.IMREAD_GRAYSCALE
    else:
      print('Unsupported image encoding type: %s.' % encoding)
      return None

  def build(self, file_name, encoding, t=None):
    pb_image = sensor_image_pb2.Image()
    flag = self._to_flag(encoding)
    if flag is None:
      return

    if t is None:
      t = time.time()

    self._build_header(pb_image.header, t)
    pb_image.frame_id = str(self._frame_id)
    pb_image.measurement_time = t
    pb_image.encoding = encoding

    img = cv2.imread(file_name, flag)

    pb_image.height, pb_image.width, channels = img.shape
    pb_image.step = pb_image.width * channels

    # todo(zero): which one is right?
    # if flag == cv2.IMREAD_COLOR:
    #   pb_image.step = pb_image.width * 3
    # elif flag == cv2.IMREAD_GRAYSCALE:
    #   pb_image.step = pb_image.width
    # else:
    #   return

    pb_image.data = img.tostring()
    self._frame_id += 1
    return pb_image


class PointCloudBuilder(Builder):
  def __init__(self) -> None:
    super().__init__()

  def build(self, file_name, t=None):
    pb_point_cloud = pointcloud_pb2.PointCloud()

    if t is None:
      t = time.time()

    self._build_header(pb_point_cloud.header, t)
    pb_point_cloud.frame_id = str(self._frame_id)
    # pb_point_cloud.is_dense = False
    pb_point_cloud.measurement_time = t

    point_cloud = pypcd.point_cloud_from_path(file_name)

    pb_point_cloud.width = point_cloud.width
    pb_point_cloud.height = point_cloud.height

    for data in point_cloud.pc_data:
      point = pb_point_cloud.point.add()
      point.x, point.y, point.z, point.intensity, timestamp = data
      point.timestamp = int(timestamp * 1e9)

    return pb_point_cloud
