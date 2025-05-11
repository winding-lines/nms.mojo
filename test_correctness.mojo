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

from gpu.host import DeviceContext
from layout.layout_tensor import Layout, LayoutTensor
from math import ceildiv
from sys import has_amd_gpu_accelerator, has_nvidia_gpu_accelerator
from operations.nms import nms_gpu
from testing import assert_equal

alias DEVICE_ID = 0


def test_nms(ctx: DeviceContext) -> None:
    alias N = 16

    alias corners_layout = Layout.row_major(N,4)
    alias flat_layout = Layout.row_major(N)

    alias float_dtype = DType.float32
    alias keep_dtype = DType.uint64

    var corners_buffer = ctx.enqueue_create_buffer[float_dtype](corners_layout.size())
    var scores_buffer = ctx.enqueue_create_buffer[float_dtype](flat_layout.size())
    var keep_buffer = ctx.enqueue_create_buffer[keep_dtype](flat_layout.size())

    with corners_buffer.map_to_host() as corners_host, scores_buffer.map_to_host() as scores_host, keep_buffer.map_to_host() as keep_host:
        var corners_tensor = LayoutTensor[float_dtype, corners_layout](corners_host)
        var scores_tensor = LayoutTensor[float_dtype, flat_layout](scores_host)
        var keep_tensor = LayoutTensor[keep_dtype, flat_layout](keep_host)
        for row in range(N):
            corners_tensor[row,0] = 0.0
            corners_tensor[row,1] = 0.0
            corners_tensor[row,2] = 1.0
            corners_tensor[row,3] = 1.0

            scores_tensor[row] = 1.0

            keep_tensor[row] = 1


    var corners_tensor = LayoutTensor[float_dtype, corners_layout](corners_buffer)
    var scores_tensor = LayoutTensor[float_dtype, flat_layout](scores_buffer)
    var keep_tensor = LayoutTensor[keep_dtype, flat_layout](keep_buffer)


    alias BN = 16
    ctx.enqueue_function[
        nms_gpu[
            float_dtype,
            corners_layout,
            flat_layout,
            flat_layout,
        ]
    ](
        corners_tensor,
        scores_tensor,
        keep_tensor,
        0.7,
        grid_dim=(ceildiv(N, BN)),
        block_dim=(BN, ),
    )

    with keep_buffer.map_to_host() as host_buffer:
        var host_tensor = LayoutTensor[keep_dtype, flat_layout](host_buffer)
        # TODO: look for a reduce function.
        var count = 0
        for i in range(N):
            if host_tensor[i] == 1:
                count += 1
        assert_equal(count, 1)


def main():
    @parameter
    if has_amd_gpu_accelerator() or has_nvidia_gpu_accelerator():
        var gpu_ctx = DeviceContext(device_id=DEVICE_ID)
        test_nms(gpu_ctx)
