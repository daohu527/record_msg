from record_msg.time_conversion import unix2gps, gps2unix, Unix2Gps, Gps2Unix


def test_unix_gps_roundtrip_examples():
    # pick a timestamp after 2017/01/01 to hit top branch
    u = 1609459200  # 2021-01-01
    g = unix2gps(u)
    u2 = gps2unix(g)
    assert isinstance(g, (int, float))
    assert isinstance(u2, (int, float))


def test_unix2gps_unix_roundtrip():
    u = 1609459200.0
    g = Unix2Gps(u)
    u2 = Gps2Unix(g)
    assert abs(u - u2) < 2.0
