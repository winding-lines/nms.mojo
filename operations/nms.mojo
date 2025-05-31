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

alias X_MIN = 0
alias Y_MIN = 1
alias X_MAX = 2
alias Y_MAX = 3

alias BBOX_LAYOUT = Layout.row_major(4)


fn iou[
        dtype: DType, f_layout: Layout, s_layout: Layout, layout_int_type: DType, linear_idx_type: DType
](
    first: LayoutTensor[
        dtype,
        f_layout,
        MutableAnyOrigin,
        layout_int_type=layout_int_type,
        linear_idx_type=linear_idx_type,
        masked=_,
    ],
    second: LayoutTensor[
        dtype,
        s_layout,
        MutableAnyOrigin,
        layout_int_type=layout_int_type,
        linear_idx_type=linear_idx_type,
        masked=_,
    ],
) -> first.element_type:
    """Compute the intersection_over_union between the 2 boxes."""
    if (
        first[0, X_MIN] > second[0, X_MAX]
        or second[0, X_MIN] > first[0, X_MAX]
        or first[0, Y_MIN] > second[0, Y_MAX]
        or second[0, Y_MIN] > first[0, Y_MAX]
    ):
        return 0

    var x_start = max(first[0, X_MIN], second[0, X_MIN])
    var x_end = min(first[0, X_MAX], second[0, X_MAX])
    var y_start = max(first[0, Y_MIN], second[0, Y_MIN])
    var y_end = min(first[0, Y_MAX], second[0, Y_MAX])

    var intersection = (x_end - x_start) * (y_end - y_start)

    var union = (first[0, X_MAX] - first[0, X_MIN]) * (
        first[0, Y_MAX] - first[0, Y_MIN]
    ) + (second[0, X_MAX] - second[0, X_MIN]) * (
        second[0, Y_MAX] - second[0, Y_MIN]
    ) - intersection

    var result = intersection / union

    return result 


fn nms[
    dtype: DType,
    corners_layout: Layout,
    score_layout: Layout,
    bitmap_layout: Layout,
](
    corners: LayoutTensor[
        dtype, corners_layout, MutableAnyOrigin
    ],  # x1, y1, x2, y2
    score: LayoutTensor[dtype, score_layout, MutableAnyOrigin],
    keep_bitmap: LayoutTensor[DType.uint8, bitmap_layout, MutableAnyOrigin],
    iou_threshold: Float32,
):
    """Process NMS on the GPU."""
    var pos = block_dim.x * block_idx.x + thread_idx.x
    var size = corners.shape[0]()
    if pos >= size:
        return

    for stride in range(1, size // 2):
        var i = pos
        var j = i + stride
        j = j % size

        if i == j:
            continue

        if keep_bitmap[i, 0] != 0 and keep_bitmap[j, 0] != 0:
            # Compute the intersection area.
            if score[i, 0] <= score[j, 0]:
                var first = corners.tile[1, 4](i, 0)
                var second = corners.tile[1, 4](j, 0)
                var overlap = iou(first, second)
                if Float32(overlap[0]) > iou_threshold:
                    # if almost equal keep the box with the higher index.
                    if abs(score[0,i]-score[0,j]) < 1e4 and i>j:
                        continue
                    keep_bitmap[i, 0] = 0
