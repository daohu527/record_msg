import os
import tempfile
import numpy as np
from modules.common_msgs.sensor_msgs import sensor_image_pb2
from record_msg.parser import ImageParser, PointCloudParser
from record_msg import pypcd


def make_pb_image(width=3, height=2, encoding='rgb8'):
    pb = sensor_image_pb2.Image()
    pb.width = width
    pb.height = height
    pb.encoding = encoding
    if encoding in ('rgb8', 'bgr8'):
        arr = np.zeros((height, width, 3), dtype=np.uint8) + 5
        pb.step = width * 3
        pb.data = arr.tobytes()
    else:
        arr = np.zeros((height, width), dtype=np.uint8) + 7
        pb.step = width
        pb.data = arr.tobytes()
    return pb


def test_image_parser_saves_and_parses(tmp_path):
    out = tmp_path
    p = ImageParser(str(out), instance_saving=True, suffix='.png')
    pb = make_pb_image()
    arr = p.parse(pb)
    # parsed array shape
    assert arr.shape == (pb.height, pb.width, 3)
    # ensure file is written after closing executor
    p.close(wait=True)
    # check for at least one file written
    files = list(out.iterdir())
    assert any(str(f).endswith('.png') for f in files)


def make_fake_pointxyzit(points):
    # Build a minimal pb pointcloud with .point repeated entries
    from modules.common_msgs.sensor_msgs.pointcloud_pb2 import PointCloud

    pc = PointCloud()
    for x, y, z, intensity, ts in points:
        pt = pc.point.add()
        pt.x = x
        pt.y = y
        pt.z = z
        pt.intensity = int(intensity)
        pt.timestamp = int(ts * 1e9)
    return pc


def test_pointcloud_parser_make_and_save(tmp_path):
    pts = [(1.0, 2.0, 3.0, 10, 1.0), (4.0, 5.0, 6.0, 20, 2.0)]
    pc_pb = make_fake_pointxyzit(pts)
    out = tmp_path
    p = PointCloudParser(str(out), instance_saving=True, suffix='.pcd')
    pc = p.parse(pc_pb, file_name='x', mode='binary_compressed')
    assert pc.points == len(pts)
    p.close(wait=True)
    files = list(out.iterdir())
    assert any(str(f).endswith('.pcd') for f in files)
