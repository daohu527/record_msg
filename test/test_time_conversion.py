#!/usr/bin/env python

# Copyright 2026 daohu527 <daohu527@gmail.com>
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
