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

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from max.driver import CPU, Device, Tensor
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, ops
from max.engine import InferenceSession


def nms(
    corners: NDArray[np.float32],
    scores: NDArray[np.float32],
    iou_threshold: float,
    session: InferenceSession,
    device: Device,
) -> Tensor:
    dtype = DType.float32

    # Create driver tensors from the input arrays, and move them to the
    # accelerator.
    corners_tensor = Tensor.from_numpy(corners).to(device)
    scores_tensor = Tensor.from_numpy(scores).to(device)

    mojo_kernels = Path(__file__).parent.parent / "operations"
    assert mojo_kernels.exists()

    # Configure our simple one-operation graph.
    with Graph(
        "nms_graph",
        input_types=[
            TensorType(
                dtype,
                shape=corners_tensor.shape,
                device=DeviceRef.from_device(device),
            ),
            TensorType(
                dtype,
                shape=scores_tensor.shape,
                device=DeviceRef.from_device(device),
            ),
        ],
        custom_extensions=[mojo_kernels],
    ) as graph:
        # Take in the two inputs to the graph.
        corners_value, scores_value = graph.inputs
        output = ops.custom(
            name="nms",
            values=[corners_value, scores_value, ops.constant(
                iou_threshold, DType.float32, DeviceRef.CPU())],
            out_types=[
                TensorType(
                    dtype=DType.uint8,
                    shape=[scores_value.tensor.shape[0]],
                    device=DeviceRef.from_device(device),
                )
            ],
            parameters={},
        )[0].tensor
        graph.output(output)

    # Compile the graph.
    print("Compiling...")
    model = session.load(graph)

    # Perform the calculation on the target device.
    print("Executing...")
    result = model.execute(corners_value, scores_value)[0]

    # Copy values back to the CPU to be read.
    assert isinstance(result, Tensor)
    return result.to(CPU())
