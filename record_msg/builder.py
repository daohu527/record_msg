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
import logging
import numpy as np

from record_msg import pypcd

from modules.drivers.proto import sensor_image_pb2, pointcloud_pb2
from modules.localization.proto import localization_pb2
from modules.transform.proto import transform_pb2

class Builder(object):
  def __init__(self) -> None:
    self._sequence_num = 0

  def _build_header(self, header,
      t=None, module_name=None, version=None, frame_id=None):
    header.sequence_num = self._sequence_num
    if t:
      header.timestamp_sec = t
      # todo(zero): no need to add?
      # header.camera_timestamp = int(t * 1e9)
      # header.lidar_timestamp = int(t * 1e9)
    if module_name:
      header.module_name = module_name
    if version:
      header.version = version
    if frame_id:
      header.frame_id = frame_id


class TransformBuilder(Builder):
  def __init__(self) -> None:
    super().__init__()

  def build(self, frame_id, child_frame_id, translation, rotation, t):
    pb_transformstampeds = transform_pb2.TransformStampeds()
    pb_transformstamped = pb_transformstampeds.transforms.add()
    if t is None:
      t = time.time()

    self._build_header(pb_transformstamped.header, t=t, frame_id=frame_id)
    pb_transformstamped.child_frame_id = child_frame_id
    pb_transformstamped.transform.translation.x = translation[0]
    pb_transformstamped.transform.translation.y = translation[1]
    pb_transformstamped.transform.translation.z = translation[2]

    pb_transformstamped.transform.rotation.qw = rotation[0]
    pb_transformstamped.transform.rotation.qx = rotation[1]
    pb_transformstamped.transform.rotation.qy = rotation[2]
    pb_transformstamped.transform.rotation.qz = rotation[3]

    self._sequence_num += 1
    return pb_transformstampeds


class LocalizationBuilder(Builder):
  def __init__(self) -> None:
    super().__init__()

  def build(self, translation, rotation, t):
    pb_localization = localization_pb2.LocalizationEstimate()
    if t is None:
      t = time.time()

    self._build_header(pb_localization.header, t=t, module_name='localization')
    pb_localization.pose.position.x = translation[0]
    pb_localization.pose.position.y = translation[1]
    pb_localization.pose.position.z = translation[2]

    pb_localization.pose.orientation.qw = rotation[0]
    pb_localization.pose.orientation.qx = rotation[1]
    pb_localization.pose.orientation.qy = rotation[2]
    pb_localization.pose.orientation.qz = rotation[3]

    # todo(zero): need to complete
    # pb_localization.pose.linear_velocity
    # pb_localization.pose.linear_acceleration
    # pb_localization.pose.angular_velocity
    # pb_localization.pose.heading

    pb_localization.measurement_time = t
    self._sequence_num += 1
    return pb_localization


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

  def build(self, file_name, frame_id, encoding, t=None):
    pb_image = sensor_image_pb2.Image()
    flag = self._to_flag(encoding)
    if flag is None:
      return

    if t is None:
      t = time.time()

    self._build_header(pb_image.header, frame_id=frame_id)
    pb_image.frame_id = frame_id
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
    self._sequence_num += 1
    return pb_image


class PointCloudBuilder(Builder):
  def __init__(self) -> None:
    super().__init__()

  def build(self, file_name, frame_id, t=None):
    pb_point_cloud = pointcloud_pb2.PointCloud()

    if t is None:
      t = time.time()

    self._build_header(pb_point_cloud.header, t=t, frame_id=frame_id)
    pb_point_cloud.frame_id = frame_id
    # pb_point_cloud.is_dense = False
    pb_point_cloud.measurement_time = t

    point_cloud = pypcd.point_cloud_from_path(file_name)

    pb_point_cloud.width = point_cloud.width
    pb_point_cloud.height = point_cloud.height

    for data in point_cloud.pc_data:
      point = pb_point_cloud.point.add()
      point.x, point.y, point.z, point.intensity, timestamp = data
      point.timestamp = int(timestamp * 1e9)

    self._sequence_num += 1
    return pb_point_cloud

  def build_nuscenes(self, file_name, frame_id, t=None):
    pb_point_cloud = pointcloud_pb2.PointCloud()

    if t is None:
      t = time.time()

    self._build_header(pb_point_cloud.header, t=t, frame_id=frame_id)
    pb_point_cloud.frame_id = frame_id
    # pb_point_cloud.is_dense = False
    pb_point_cloud.measurement_time = t

    # Loads LIDAR data from binary numpy format.
    # Data is stored as (x, y, z, intensity, ring index).
    scan = np.fromfile(file_name, dtype=np.float32)
    logging.debug(scan[:100])

    points = scan.reshape((-1, 5))[:, :4]
    logging.debug("points: {},{}".format(np.shape(points), points.dtype))

    pb_point_cloud.width = len(points)
    pb_point_cloud.height = 1

    # Points shape is (length, 4)
    n0, _ = np.shape(points)
    for i in range(n0):
      point = pb_point_cloud.point.add()
      point.x, point.y, point.z, intensity = points[i]
      point.intensity = int(intensity)
    self._sequence_num += 1
    return pb_point_cloud
