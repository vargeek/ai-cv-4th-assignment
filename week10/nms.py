import functools

def area(roi):
    """
    area  
    roi: [x1,x2,y1,y2]  
    """
    return (roi[1] - roi[0]) * (roi[3] - roi[2])

def IOU(roi1, roi2):
    """
    intersection over union  
    ori1: [x1,x2,y1,y2]  
    ori2: [x1,x2,y1,y2]  
    """
    left = roi1 if roi1[0] < roi2[0] else roi2
    right = roi2 if roi1[0] < roi2[0] else roi1
    top = roi1 if roi1[2] < roi2[2] else roi2
    bottom = roi2 if roi1[2] < roi2[2] else roi1
    if right[0] > left[1] or bottom[2] > top[3]:
        return 0

    w = left[1] - right[0]
    h = top[3] - bottom[2]

    intersection = w * h
    union = area(roi1) + area(roi2) - intersection

    return intersection / union


def NMS(lists, threshold):
    """
    Non-Maximum Suppression  
    lists: lists[0:4]: x1, x2, y1, y2; lists[4]: score
    """
    results = []

    def sort_by_score(x, y):
        return -1 if x[4] > y[4] else 1

    lists.sort(key=functools.cmp_to_key(sort_by_score))
    
    i = 0
    while i < len(lists):
        tmp = lists[i]
        i = i + 1
        results.append(tmp)

        for j in range(i, len(lists)):
            if IOU(tmp, lists[j]) > threshold:
                lists[i], lists[j] = lists[j], lists[i] 
                i = i + 1

    return results
