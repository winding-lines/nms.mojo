# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

import numpy as np
from max.driver import Accelerator, CPU, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from .common import nms


def test_nms(session: InferenceSession) -> None:
    N = 256

    corners = np.random.uniform(size=(N, 4)).astype(np.float32)
    scores = np.random.uniform(size=(N)).astype(np.float32)
    expected_result = np.random.uniform(size=(N)).astype(np.uint8)

    result = nms(corners, scores, 0.7, session, session.devices[0])

    assert np.all(np.isclose(result.to_numpy(), expected_result))
    assert result.dtype == DType.uint8
    assert result.shape == (N,)
