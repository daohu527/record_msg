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


from PIL import Image
import numpy as np
import os
import threading
import logging
from concurrent.futures import ThreadPoolExecutor

from functools import reduce

from record_msg import pypcd


LOGGER = logging.getLogger(__name__)


def to_csv(msg):
    """Flatten the properties of the object, output as an array

    Args:
        msg (_type_): _description_

    Returns:
        _type_: _description_
    """
    if isinstance(msg, (int, float, bool, str, bytes)):
        return [msg]
    elif msg and isinstance(msg, (tuple, list)):
        flat = []
        for m in msg:
            flat.extend(to_csv(m))
        return flat
    elif hasattr(msg, "DESCRIPTOR"):
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
        # Ensure output dir exists to avoid errors when saving
        try:
            if self._output_path:
                os.makedirs(self._output_path, exist_ok=True)
        except Exception as exc:
            LOGGER.warning(
                "Failed to create output directory '%s': %s", self._output_path, exc
            )
        # Executor for background saving tasks
        self._save_executor = ThreadPoolExecutor(max_workers=2)
        # Lock for thread-safe increments
        self._msg_lock = threading.Lock()
        self._closed = False

    def close(self, wait=True):
        """Shut down background resources used by the parser.

        Call this when the parser is no longer needed to avoid leaking threads.
        Supports `wait=True` to block until running tasks finish.
        """
        if getattr(self, "_save_executor", None) is not None:
            try:
                self._save_executor.shutdown(wait=wait)
            except Exception as exc:
                LOGGER.warning("Failed to shutdown save executor cleanly: %s", exc)
            finally:
                self._save_executor = None
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close(wait=True)


class ImageParser(Parser):
    def __init__(self, output_path, instance_saving=True, suffix=".jpg") -> None:
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
        if image.encoding == "rgb8" or image.encoding == "bgr8":
            if image.step != image.width * 3:
                print(
                    "Image.step %d does not equal to Image.width %d * 3 for color image."
                    % (image.step, image.width)
                )
                return False
            else:
                return True
        elif image.encoding == "gray" or image.encoding == "y":
            if image.step != image.width:
                print(
                    "Image.step %d does not equal to Image.width %d or gray image."
                    % (image.step, image.width)
                )
                return False
            else:
                return True
        else:
            print("Unsupported image encoding type: %s." % image.encoding)
            return False

    def parse(self, image, file_name=None):
        if not self._valid(image):
            return None

        # Use frombuffer (no-deprecation) and reshape to (H,W,C)
        channel_num = image.step // image.width
        self._parsed_data = np.frombuffer(image.data, dtype=np.uint8)
        try:
            self._parsed_data = self._parsed_data.reshape(
                (image.height, image.width, channel_num)
            )
        except Exception:
            # fallback to empty array if data size mismatches
            self._parsed_data = np.zeros(
                (image.height, image.width, channel_num), dtype=np.uint8
            )
        self._encoding = image.encoding

        if self._instance_saving:
            if file_name is None:
                with self._msg_lock:
                    file_name = "%05d" % self._msg_count + self._suffix
                    self._msg_count += 1
            else:
                file_name = str(file_name) + self._suffix
            output_file = os.path.join(self._output_path, file_name)
            # save in background to avoid blocking parsing
            try:
                self.save_image_mat_to_file(
                    image_file=output_file,
                    image_mat=self._parsed_data,
                    encoding=self._encoding,
                )
            except Exception:
                # fallback to synchronous save on unexpected errors
                self._sync_save_image(
                    image_mat=self._parsed_data,
                    encoding=self._encoding,
                    image_file=output_file,
                )
        return self._parsed_data

    def save_image_mat_to_file(self, image_file, image_mat=None, encoding=None):
        # Save image in BGR oder
        if image_mat is None:
            image_mat = self._parsed_data
        if encoding is None:
            encoding = self._encoding

        def _do_save():
            # Use Pillow for saving; convert channel order when needed
            if encoding == "rgb8":
                # Pillow expects RGB order
                im = Image.fromarray(image_mat, mode="RGB")
            elif encoding == "bgr8":
                # convert BGR -> RGB
                im = Image.fromarray(image_mat[..., ::-1], mode="RGB")
            elif encoding in ("gray", "y"):
                # ensure 2D array for L mode
                if image_mat.ndim == 3 and image_mat.shape[2] == 1:
                    arr = image_mat[:, :, 0]
                else:
                    arr = image_mat
                im = Image.fromarray(arr, mode="L")
            else:
                im = Image.fromarray(image_mat)
            im.save(image_file)

        # Submit to executor to avoid blocking; keep synchronous fallback
        if hasattr(self, "_save_executor") and self._save_executor is not None:
            self._save_executor.submit(_do_save)
        else:
            _do_save()

    def _sync_save_image(self, image_mat, encoding, image_file):
        if encoding == "rgb8":
            im = Image.fromarray(image_mat, mode="RGB")
        elif encoding == "bgr8":
            im = Image.fromarray(image_mat[..., ::-1], mode="RGB")
        elif encoding in ("gray", "y"):
            if image_mat.ndim == 3 and image_mat.shape[2] == 1:
                arr = image_mat[:, :, 0]
            else:
                arr = image_mat
            im = Image.fromarray(arr, mode="L")
        else:
            im = Image.fromarray(image_mat)
        im.save(image_file)


class PointCloudParser(Parser):
    def __init__(self, output_path, instance_saving=True, suffix=".pcd"):
        super(PointCloudParser, self).__init__(output_path, instance_saving, suffix)
        # Cache dtype/typenames for repeated pointcloud conversions
        if not hasattr(PointCloudParser, "_cached_np_dtype"):
            PointCloudParser._cached_np_dtype = None

    def convert_xyzit_pb_to_array(self, xyz_i_t, data_type):
        n = len(xyz_i_t)
        if n == 0:
            return np.zeros(0, dtype=data_type)
        # Use fromiter/array from list of tuples to avoid per-index assignment
        gen = ((p.x, p.y, p.z, p.intensity, p.timestamp / 1e9) for p in xyz_i_t)
        try:
            return np.fromiter(gen, dtype=data_type, count=n)
        except Exception:
            # fallback: build list then array
            return np.array(
                list(
                    ((p.x, p.y, p.z, p.intensity, p.timestamp / 1e9) for p in xyz_i_t)
                ),
                dtype=data_type,
            )

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

        md = {
            "version": 0.7,
            "fields": ["x", "y", "z", "intensity", "timestamp"],
            "count": [1, 1, 1, 1, 1],
            "width": len(xyz_i_t),
            "height": 1,
            "viewpoint": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "points": len(xyz_i_t),
            "type": ["F", "F", "F", "U", "F"],
            "size": [4, 4, 4, 4, 8],
            "data": "binary_compressed",
        }

        typenames = []
        # Compute and cache dtype since type/size mapping is constant
        if PointCloudParser._cached_np_dtype is None:
            for t, s in zip(md["type"], md["size"]):
                np_type = pypcd.pcd_type_to_numpy_type[(t, s)]
                typenames.append(np_type)
            PointCloudParser._cached_np_dtype = np.dtype(
                list(zip(md["fields"], typenames))
            )

        np_dtype = PointCloudParser._cached_np_dtype
        pc_data = self.convert_xyzit_pb_to_array(xyz_i_t, data_type=np_dtype)
        pc = pypcd.PointCloud(md, pc_data)
        return pc

    def save_pointcloud_meta_to_file(self, pc_meta, pcd_file, mode):
        def _do_save():
            if mode == "ascii":
                pypcd.save_point_cloud(pc_meta, pcd_file)
            elif mode == "binary":
                pypcd.save_point_cloud_bin(pc_meta, pcd_file)
            elif mode == "binary_compressed":
                pypcd.save_point_cloud_bin_compressed(pc_meta, pcd_file)
            else:
                print("Unknown point cloud format!")

        if hasattr(self, "_save_executor") and self._save_executor is not None:
            self._save_executor.submit(_do_save)
        else:
            _do_save()

    def parse(self, pointcloud, file_name=None, mode="ascii"):
        """
        Transform protobuf PointXYZIT to standard PCL bin_compressed_file(*.pcd).
        """
        self._parsed_data = self.make_xyzit_point_cloud(pointcloud.point)

        if self._instance_saving:
            if file_name is None:
                with self._msg_lock:
                    file_name = "%05d" % self._msg_count + self._suffix
                    self._msg_count += 1
            else:
                file_name = str(file_name) + self._suffix
            output_file = os.path.join(self._output_path, file_name)
            try:
                self.save_pointcloud_meta_to_file(
                    pc_meta=self._parsed_data, pcd_file=output_file, mode=mode
                )
            except Exception:
                # synchronous fallback
                if mode == "ascii":
                    pypcd.save_point_cloud(self._parsed_data, output_file)
                elif mode == "binary":
                    pypcd.save_point_cloud_bin(self._parsed_data, output_file)
                elif mode == "binary_compressed":
                    pypcd.save_point_cloud_bin_compressed(
                        self._parsed_data, output_file
                    )
                else:
                    print("Unknown point cloud format!")
        return self._parsed_data
