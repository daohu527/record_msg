import os
import pytest

from record_msg.builder import ImageBuilder, PointCloudBuilder


@pytest.mark.integration
def test_integration_image_builder_reads_sample():
    here = os.path.dirname(__file__)
    img_path = os.path.join(here, 'test.jpg')
    assert os.path.exists(img_path), f"sample image not found: {img_path}"
    b = ImageBuilder()
    pb_image = b.build(img_path, frame_id='f_integ', encoding='rgb8')
    assert pb_image is not None
    # basic sanity checks
    assert getattr(pb_image, 'width', 0) > 0
    assert getattr(pb_image, 'height', 0) > 0


@pytest.mark.integration
def test_integration_point_cloud_builder_reads_sample():
    here = os.path.dirname(__file__)
    pcd_path = os.path.join(here, 'test.pcd')
    assert os.path.exists(pcd_path), f"sample pcd not found: {pcd_path}"
    b = PointCloudBuilder()
    pb_pc = b.build(pcd_path, frame_id='pc_integ')
    assert pb_pc is not None
    # width/height should be positive for a valid pcd
    assert getattr(pb_pc, 'width', 0) > 0
    assert getattr(pb_pc, 'height', 0) >= 1
