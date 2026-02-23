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


import numpy as np


def transform_mat(pc_data, T):
    return pc_data @ T


def transform_points(pc_data, T):
    return np.vstack([point @ T for point in pc_data])


def test_transform_equivalence():
    np.random.seed(0)
    # small synthetic point set (n,4)
    pc = np.random.randn(10, 4)
    T_id = np.eye(4)
    T_rand = np.random.randn(4, 4)

    assert np.allclose(transform_mat(pc, T_id), transform_points(pc, T_id))
    assert np.allclose(transform_mat(pc, T_rand), transform_points(pc, T_rand))


def test_transform_handles_empty():
    pc = np.zeros((0, 4))
    T = np.eye(4)
    assert transform_mat(pc, T).shape[0] == 0
