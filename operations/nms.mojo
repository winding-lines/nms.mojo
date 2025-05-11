from gpu import WARP_SIZE, barrier, block_dim, block_idx, thread_idx
from gpu.memory import async_copy_wait_all
from layout.layout_tensor import (
    Layout,
    LayoutTensor,
    copy_dram_to_sram,
    copy_dram_to_sram_async,
)
from layout.math import outer_product_acc
from layout.tensor_builder import LayoutTensorBuild as tb
from tensor import InputTensor, ManagedTensorSlice, OutputTensor
from sys.info import simdwidthof


fn _iou[
    type: DType
](
    ax1: Scalar[type],
    ay1: Scalar[type],
    ax2: Scalar[type],
    ay2: Scalar[type],
    bx1: Scalar[type],
    by1: Scalar[type],
    bx2: Scalar[type],
    by2: Scalar[type],
) -> Scalar[type]:
    """Compute the intersection_over_union between the 2 boxes."""
    if ax1 > bx2 or bx1 > ax2 or ay1 > by2 or by1 > ay2:
        return 0

    var x_start = max(ax1, bx1)
    var x_end = min(ax2, bx2)
    var y_start = max(ay1, by1)
    var y_end = min(ay2, by2)

    var intersection = (x_end - x_start) * (y_end - y_start)

    var union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (
        by2 - by1
    ) - intersection

    return intersection / union


fn nms_gpu[
    dtype: DType,
    corners_layout: Layout,
    score_layout: Layout,
    bitmap_layout: Layout,
](
    corners: LayoutTensor[
        dtype, corners_layout, MutableAnyOrigin
    ],  # x1, y1, x2, y2
    score: LayoutTensor[dtype, score_layout, MutableAnyOrigin],
    keep_bitmap: LayoutTensor[dtype, bitmap_layout, MutableAnyOrigin],
    iou_threshold: Scalar[dtype],
):
    """Process NMS on the GPU."""
    var pos = block_dim.x * block_idx.x + thread_idx.x
    var size = corners.shape[0]()

    for stride in range(size // 2):
        var i = (stride + 1) * pos
        var j = (stride + 1) * pos + pos - 1
        j = j % size

        if ( keep_bitmap[i] != 0 and keep_bitmap[j] != 0):
            # Compute the intersection area.
            if (
                score[i] > score[j]
                and _iou(
                    corners[i][0],
                    corners[i][1],
                    corners[i][2],
                    corners[i][3],
                    corners[j][0],
                    corners[j][1],
                    corners[j][2],
                    corners[j][3],
                )
                > iou_threshold
            ):
                keep_bitmap[j] = 0
