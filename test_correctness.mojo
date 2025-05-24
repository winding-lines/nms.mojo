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
from operations.nms import nms, iou
from testing import assert_equal
from gpu.host import DeviceBuffer

alias DEVICE_ID = 0


def test_nms(ctx: DeviceContext) -> None:
    alias N = 4

    alias corners_layout = Layout.row_major(N, 4)
    alias flat_layout = Layout.row_major(N, 1)

    alias float_dtype = DType.float32
    alias keep_dtype = DType.uint8

    var corners_buffer = ctx.enqueue_create_buffer[float_dtype](
        corners_layout.size()
    )
    var scores_buffer = ctx.enqueue_create_buffer[float_dtype](
        flat_layout.size()
    )
    var keep_buffer = ctx.enqueue_create_buffer[keep_dtype](flat_layout.size())

    with corners_buffer.map_to_host() as corners_host, scores_buffer.map_to_host() as scores_host, keep_buffer.map_to_host() as keep_host:
        var corners_tensor = LayoutTensor[float_dtype, corners_layout](
            corners_host
        )
        var scores_tensor = LayoutTensor[float_dtype, flat_layout](scores_host)
        var keep_tensor = LayoutTensor[keep_dtype, flat_layout](keep_host)
        for row in range(N):
            # Setup all the bounding boxes to overlap.
            corners_tensor[row, 0] = 0.0
            corners_tensor[row, 1] = 0.0
            corners_tensor[row, 2] = 1.0
            corners_tensor[row, 3] = 1.0

            scores_tensor[row, 0] = 1.0

            keep_tensor[row, 0] = 1

        print("test: keep_tensor cpu", keep_tensor)

    var corners_tensor = LayoutTensor[float_dtype, corners_layout](
        corners_buffer
    )
    var scores_tensor = LayoutTensor[float_dtype, flat_layout](scores_buffer)
    var keep_tensor = LayoutTensor[keep_dtype, flat_layout](keep_buffer)

    alias BN = 2
    ctx.enqueue_function[
        nms[
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
        grid_dim=(
            1,
            1,
        ),  # Not really used at the moment but the compiler blows up if I don't pass them in.
        block_dim=(256, 1),  # Same as above.
    )
    ctx.synchronize()

    with keep_buffer.map_to_host() as host_buffer:
        var host_tensor = LayoutTensor[keep_dtype, flat_layout](host_buffer)
        print("test: after call, on the cpu", host_buffer)
        var count = 0
        for i in range(N):
            if host_tensor[i, 0] == 1:
                count += 1
        # We should have kept just one bounding box.
        assert_equal(count, 1)


# def test_iou(ctx: DeviceContext) -> None:
#     alias N = 2
#
#     alias corners_layout = Layout.row_major(N, 4)
#
#     alias float_dtype = DType.float32
#
#     var corners_buffer = ctx.enqueue_create_buffer[float_dtype](
#         corners_layout.size()
#     )
#
#     with corners_buffer.map_to_host() as corners_host:
#         var corners_tensor = LayoutTensor[float_dtype, corners_layout, layout_int_type=DType.uint32](
#             corners_host
#         )
#         for row in range(N):
#             # Setup all the bounding boxes to overlap.
#             corners_tensor[row, 0] = 0.0 
#             corners_tensor[row, 1] = 0.0
#             corners_tensor[row, 2] = 1.0
#             corners_tensor[row, 3] = 1.0
#
#
#     var corners_tensor = LayoutTensor[float_dtype, corners_layout](
#         corners_buffer
#     )
#
#     var first = corners_tensor.tile[1,4](0,0)
#     var second = corners_tensor.tile[1,4](1,0)
#     var overlap = iou[layout_int_type=DType.uint32, linear_idx_type=DType.uint32](first, second)
#     assert_equal(overlap, 1.0)


def main():

    @parameter
    if has_amd_gpu_accelerator() or has_nvidia_gpu_accelerator():
        var gpu_ctx = DeviceContext(device_id=DEVICE_ID)
        # test_iou(gpu_ctx)
        test_nms(gpu_ctx)
