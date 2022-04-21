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

###############################################################################

# Copyright 2019 The Apollo Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import cv2
import numpy as np
import os

from functools import reduce

from record_msg import pypcd


def to_csv(msg):
  """Flatten the properties of the object, output as an array

  Args:
      msg (_type_): _description_

  Returns:
      _type_: _description_
  """
  if type(msg) in (int, float, bool, str, bytes):
    return [msg]
  elif msg and type(msg) in (tuple, list):
    return reduce(lambda x, y: x + y, map(lambda m: to_csv(m), msg))
  elif hasattr(msg, 'DESCRIPTOR'):
    pb_attrs = msg.DESCRIPTOR.fields_by_name.keys()
    flat_attrs = list(map(lambda attr: getattr(msg, attr), pb_attrs))
    return to_csv(flat_attrs)
  else:
    return []


class Parser(object):
  def __init__(self, output_path, instance_saving, suffix):
    self._output_path = output_path
    self._instance_saving = instance_saving
    self._suffix = suffix
    self._msg_count = 0


class ImageParser(Parser):
  def __init__(self, output_path, instance_saving=True, suffix='.jpg') -> None:
    super(ImageParser, self).__init__(output_path, instance_saving, suffix)

  def _valid(self, image):
    # Save image according to cyber format, defined in sensor camera proto.
    # height = 4, image height, that is, number of rows.
    # width = 5,  image width, that is, number of columns.
    # encoding = 6, as string, type is 'rgb8', 'bgr8' or 'gray'.
    # step = 7, full row length in bytes.
    # data = 8, actual matrix data in bytes, size is (step * rows).
    # type = CV_8UC1 if image step is equal to width as gray, CV_8UC3
    # if step * 3 is equal to width.
    if image.encoding == 'rgb8' or image.encoding == 'bgr8':
      if image.step != image.width * 3:
        print('Image.step %d does not equal to Image.width %d * 3 for color image.'
              % (image.step, image.width))
        return False
      else:
        return True
    elif image.encoding == 'gray' or image.encoding == 'y':
      if image.step != image.width:
        print('Image.step %d does not equal to Image.width %d or gray image.'
              % (image.step, image.width))
        return False
      else:
        return True
    else:
      print('Unsupported image encoding type: %s.' % image.encoding)
      return False

  def parse(self, image, file_name=None):
    if not self._valid(image):
      return None

    channel_num = image.step // image.width
    self._parsed_data = np.fromstring(image.data, dtype=np.uint8).reshape(
        (image.height, image.width, channel_num))
    self._encoding = image.encoding

    if self._instance_saving:
      if file_name is None:
        file_name = "%05d" % self._msg_count + self._suffix
      else:
        file_name = str(file_name) + self._suffix
      output_file = os.path.join(self._output_path, file_name)
      self.save_image_mat_to_file(image_file=output_file)
      self._msg_count += 1
    return self._parsed_data

  def save_image_mat_to_file(self, image_file):
    # Save image in BGR oder
    image_mat = self._parsed_data
    if self._encoding == 'rgb8':
      cv2.imwrite(image_file, cv2.cvtColor(image_mat, cv2.COLOR_RGB2BGR))
    else:
      cv2.imwrite(image_file, image_mat)


class PointCloudParser(Parser):
  def __init__(self, output_path, instance_saving=True, suffix='.pcd'):
    super(PointCloudParser, self).__init__(output_path, instance_saving, suffix)

  def convert_xyzit_pb_to_array(self, xyz_i_t, data_type):
    arr = np.zeros(len(xyz_i_t), dtype=data_type)
    for i, point in enumerate(xyz_i_t):
      # change timestamp to timestamp_sec
      arr[i] = (point.x, point.y, point.z,
                point.intensity, point.timestamp/1e9)
    return arr

  def make_xyzit_point_cloud(self, xyz_i_t):
    """
    Make a pointcloud object from PointXYZIT message, as Pointcloud.proto.
    message PointXYZIT {
      optional float x = 1 [default = nan];
      optional float y = 2 [default = nan];
      optional float z = 3 [default = nan];
      optional uint32 intensity = 4 [default = 0];
      optional uint64 timestamp = 5 [default = 0];
    }
    """

    md = {'version': .7,
          'fields': ['x', 'y', 'z', 'intensity', 'timestamp'],
          'count': [1, 1, 1, 1, 1],
          'width': len(xyz_i_t),
          'height': 1,
          'viewpoint': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
          'points': len(xyz_i_t),
          'type': ['F', 'F', 'F', 'U', 'F'],
          'size': [4, 4, 4, 4, 8],
          'data': 'binary_compressed'}

    typenames = []
    for t, s in zip(md['type'], md['size']):
      np_type = pypcd.pcd_type_to_numpy_type[(t, s)]
      typenames.append(np_type)

    np_dtype = np.dtype(list(zip(md['fields'], typenames)))
    pc_data = self.convert_xyzit_pb_to_array(xyz_i_t, data_type=np_dtype)
    pc = pypcd.PointCloud(md, pc_data)
    return pc

  def save_pointcloud_meta_to_file(self, pc_meta, pcd_file, mode):
    if mode == 'ascii':
      pypcd.save_point_cloud(pc_meta, pcd_file)
    elif mode == 'binary':
      pypcd.save_point_cloud_bin(pc_meta, pcd_file)
    elif mode == 'binary_compressed':
      pypcd.save_point_cloud_bin_compressed(pc_meta, pcd_file)
    else:
      print("Unknown point cloud format!")

  def parse(self, pointcloud, file_name=None, mode='ascii'):
    """
    Transform protobuf PointXYZIT to standard PCL bin_compressed_file(*.pcd).
    """
    self._parsed_data = self.make_xyzit_point_cloud(pointcloud.point)

    if self._instance_saving:
      if file_name is None:
        file_name = "%05d" % self._msg_count + self._suffix
      else:
        file_name = str(file_name) + self._suffix
      output_file = os.path.join(self._output_path, file_name)
      self.save_pointcloud_meta_to_file(pc_meta=self._parsed_data, \
          pcd_file=output_file, mode=mode)
      self._msg_count += 1
    return self._parsed_data
