import time
from record_msg.builder import GnssBestPoseBuilder


def test_gnss_builder_required_fields():
    b = GnssBestPoseBuilder()
    pb = b.build(latitude=10.0, longitude=20.0, height_msl=100.0, undulation=30.0, t=1600000000.0)
    assert pb.latitude == 10.0
    assert pb.longitude == 20.0
    assert pb.height_msl == 100.0
    assert pb.undulation == 30.0
    assert hasattr(pb, 'measurement_time')


def test_gnss_builder_with_kwargs_and_types():
    b = GnssBestPoseBuilder()
    pb = b.build(0.0, 0.0, 0.0, 0.0, t=time.time(), status=3)
    # status is optional; protobuf may or may not have it depending on proto
    # ensure builder did not raise and returns a message
    assert pb is not None


def test_gnss_builder_input_validation():
    b = GnssBestPoseBuilder()
    try:
        b.build('not-a-number', 0.0, 0.0, 0.0)
        assert False, 'expected TypeError'
    except TypeError:
        pass
    try:
        b.build(1000.0, 0.0, 0.0, 0.0)
        assert False, 'expected ValueError for latitude range'
    except ValueError:
        pass
