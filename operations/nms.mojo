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
    print("iou: entering")
    if (
        first[X_MIN] > second[X_MAX]
        or second[X_MIN] > first[X_MAX]
        or first[Y_MIN] > second[Y_MAX]
        or second[Y_MIN] > first[Y_MAX]
    ):
        print("iou: non overlap, returning 0")
        return 0

    var x_start = max(first[X_MIN], second[X_MIN])
    var x_end = min(first[X_MAX], second[X_MAX])
    var y_start = max(first[Y_MIN], second[Y_MIN])
    var y_end = min(first[Y_MAX], second[Y_MAX])

    var intersection = (x_end - x_start) * (y_end - y_start)

    var union = (first[X_MAX] - first[X_MIN]) * (
        first[Y_MAX] - first[Y_MIN]
    ) + (second[X_MAX] - second[X_MIN]) * (
        second[Y_MAX] - second[Y_MIN]
    ) - intersection

    var result = intersection / union

    print("iou: overlap ", result)
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
    iou_threshold: Scalar[dtype],
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

        if pos == 0:
            print(
                "nms: pos ",
                pos,
                "stride",
                stride,
                "i",
                i,
                "keep[i]",
                keep_bitmap[i, 0],
                "score",
                score[i, 0],
                "  j",
                j,
                "keep[j]",
                keep_bitmap[j, 0],
                "score[j]",
                score[j, 0],
                " score i > j",
                score[i, 0] >= score[j, 0]
            )
        if keep_bitmap[i, 0] != 0 and keep_bitmap[j, 0] != 0:
            # Compute the intersection area.
            if score[i, 0] >= score[j, 0]:
                var first = corners.tile[1, 4](i, 0)
                var second = corners.tile[1, 4](j, 0)
                print("nms: calling iou")
                var overlap = iou(first, second)
                print("nms: iou returned overlap", overlap)
                if overlap > iou_threshold:
                    keep_bitmap[j, 0] = 0
                    print("nms: discarding box", j)
