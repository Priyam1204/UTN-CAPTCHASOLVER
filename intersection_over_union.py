def intersect_over_union(GT_bbox, PD_bbox):
    gt_x = max(GT_bbox[0], PD_bbox[0])
    gt_y = max(GT_bbox[1], PD_bbox[1])
    pd_x = min(GT_bbox[2], PD_bbox[2])
    pd_y = min(GT_bbox[3], PD_bbox[3])

    # intersection area
    inter_w = max(0, pd_x - gt_x)
    inter_h = max(0, pd_y - gt_y)
    intersection_area = inter_w * inter_h

    # areas of the boxes
    GT_BoxArea = max(0, GT_bbox[2] - GT_bbox[0]) * max(0, GT_bbox[3] - GT_bbox[1])
    PD_BoxArea = max(0, PD_bbox[2] - PD_bbox[0]) * max(0, PD_bbox[3] - PD_bbox[1])

    # IoU
    union = GT_BoxArea + PD_BoxArea - intersection_area
    intersection_over_union = intersection_area / float(union + 1e-9)
    return intersection_over_union
