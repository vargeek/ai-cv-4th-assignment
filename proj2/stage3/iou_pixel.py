def area(roi):
    """
    area  
    :param roi: [x_tl, y_tl, x_br, y_br]
    """
    return (roi[2] - roi[0] + 1) * (roi[3] - roi[1] + 1)

def IOU(roi1, roi2):
    """
    Intersection over Union
    :param roi1: [x_tl, y_tl, x_br, y_br]
    :param roi2: [x_tl, y_tl, x_br, y_br]
    """
    left = max(roi1[0], roi2[0])
    right = min(roi1[2], roi2[2])
    top = max(roi1[1], roi2[1])
    bottom = min(roi1[3], roi2[3])

    width = right - left + 1 if left <= right else 0
    height = bottom - top + 1 if top <= bottom else 0

    intersection_area = width * height
    union_area = area(roi1) + area(roi2) - intersection_area
    return 1.0 * intersection_area / union_area if union_area > 0 else 0
