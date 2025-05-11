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

import compiler
from gpu.host import DeviceBuffer
from math import ceildiv
from memory import UnsafePointer
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor
from .nms import nms_gpu

@compiler.register("non_maximum_supression")
struct NonMaximumSupression:
    """
    The custom operation for NMS.
    """

    @staticmethod
    fn execute[
        # The kind of device this will be run on: "cpu" or "gpu"
        target: StaticString,
    ](
        keep_bitmap: OutputTensor[rank=1],
        corners: InputTensor[rank = keep_bitmap.rank],
        scores: InputTensor[rank = keep_bitmap.rank],
        iou_threshold: InputTensor,
        # the context is needed for some GPU calls
        ctx: DeviceContextPtr,
    ) raises:
        corners_layout = corners.to_layout_tensor()
        scores_layout = scores.to_layout_tensor()
        bitmap_layout = keep_bitmap.to_layout_tensor()

        size = corners_layout.shape[0]()

        gpu_ctx = ctx.get_device_context()

        # Initialize the keep_bitmap.
        gpu_ctx.enqueue_memset(
            DeviceBuffer[keep_bitmap.type](
                gpu_ctx,
                rebind[UnsafePointer[Scalar[keep_bitmap.type]]](bitmap_layout.ptr),
                size,
                owning=False,
            ),
            1,
        )

        alias BN = 32
        gpu_ctx.enqueue_function[
            nms_gpu[
                corners.type,
                corners_layout.layout,
                scores_layout.layout,
                bitmap_layout.layout,
            ]
        ](
            corners_layout,
            scores_layout,
            bitmap_layout,
            0.7,
             )

