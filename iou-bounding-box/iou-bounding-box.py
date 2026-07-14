def iou(box_a, box_b):
    """
    Compute Intersection over Union of two bounding boxes.
    """
    x1_a, y1_a, x2_a, y2_a = box_a
    x1_b, y1_b, x2_b, y2_b = box_b

    # compute area of A, B
    area_a = max(0.0, x2_a - x1_a) * max(0.0, y2_a - y1_a)
    area_b = max(0.0, x2_b - x1_b) * max(0.0, y2_b - y1_b)

    # find intersection coordinate
    x1_i = max(x1_a, x1_b)
    y1_i = max(y1_a, y1_b)
    x2_i = min(x2_a, x2_b)
    y2_i = min(y2_a, y2_b)

    # compute the intersection area
    inter_w = max(0.0, x2_i - x1_i)
    inter_h = max(0.0, y2_i - y1_i)
    intersection = inter_w * inter_h

    # compute the union area
    union = area_a + area_b - intersection

    # edge area = 0 handle
    if union == 0.0:
        return 0.0

    return float(intersection/union)