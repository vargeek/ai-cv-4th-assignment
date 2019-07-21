import numpy as np
import os
import sys
import cv2


def medianBlur(img, kernel, padding_way):
    """
    img & kernel is List of List
    padding_way a string
    """

    index_fn = padding_way.upper() == 'REPLICA' and replicaPaddingIndex or zeroPaddingIndex
    rows, cols, *_ = img.shape
    krows, kcols, *_ = kernel.shape
    ksize = krows * kcols
    row_offset = -(krows // 2)
    col_offset = -(kcols // 2)

    pixels = np.zeros((ksize,), dtype=img.dtype)
    result = np.zeros((rows, cols), dtype=img.dtype)

    for row in range(0, rows):
        for col in range(0, cols):
            row_start = row + row_offset
            col_start = col + col_offset
            idx = 0
            for r in range(row_start, row_start + krows):
                for c in range(col_start, col_start + kcols):
                    pixels[idx] = index_fn(img, r, c)
                    idx = idx + 1
            pixels.sort()
            # quickSort(pixels)
            result[row][col] = pixels[(ksize-1)//2]
    return result


def medianBlurQuickSelect(img, kernel, padding_way):
    """
    medianBlur(使用QuickSelect算法查找中位数)
    img & kernel is List of List
    padding_way a string
    """

    index_fn = padding_way.upper() == 'REPLICA' and replicaPaddingIndex or zeroPaddingIndex
    rows, cols, *_ = img.shape
    krows, kcols, *_ = kernel.shape
    ksize = krows * kcols
    row_offset = -(krows // 2)
    col_offset = -(kcols // 2)

    pixels = np.zeros((ksize,), dtype=img.dtype)
    result = np.zeros((rows, cols), dtype=img.dtype)

    for row in range(0, rows):
        for col in range(0, cols):
            row_start = row + row_offset
            col_start = col + col_offset
            idx = 0
            for r in range(row_start, row_start + krows):
                for c in range(col_start, col_start + kcols):
                    pixels[idx] = index_fn(img, r, c)
                    idx = idx + 1

            result[row][col] = quickSelectMedianValue(pixels)
    return result


def medianBlurHistogram(img, kernel, padding_way):
    """
    medianBlur(使用灰度值直方图统计)
    img & kernel is List of List
    padding_way a string
    """
    rows, cols, *_ = img.shape  # 图片行数、列数
    krows, kcols, *_ = kernel.shape  # kernel行数、列数
    pre_rows = krows // 2  # kernel在当前点上面的行数
    post_rows = (krows - 1) // 2  # kernel在当前点下面的行数
    pre_cols = kcols // 2  # kernel在当前点左边的列数
    post_cols = (kcols - 1) // 2  # kernel当前点右边的列数
    middle = (krows * kcols + 1) // 2
    index_fn = padding_way.upper() == 'REPLICA' and replicaPaddingIndex or zeroPaddingIndex

    # 每列的直方图, 256 bin
    hist_of_cols = np.zeros((cols + 2, 256), dtype=np.int)
    # 当前kernel直方图, 256 bin
    hist_of_kernel = np.zeros((256,), dtype=np.int)

    # 初始化每列直方图
    for col in range(-1, cols + 1):
        pixel = index_fn(img, -1, col)
        # 多加一是因为更新`hist_of_kernel`会再减一次
        hist_of_col = hist_of_cols[1+col]
        hist_of_col[pixel] = hist_of_col[pixel] + pre_rows + 1
        for row in range(0, post_rows):
            pixel = index_fn(img, row, col)
            hist_of_col[pixel] = hist_of_col[pixel] + 1

    # 过滤后的图像
    result = np.zeros((rows, cols), dtype=img.dtype)
    for row in range(0, rows):
        # 初始化当前kernel直方图
        hist_of_kernel[:] = 0
        # 更新前 `post_cols+1` 列直方图
        for col in range(-1, post_cols):
            hist_of_col = hist_of_cols[col+1]
            top_pixel = index_fn(img, row-pre_rows-1, col)
            bottom_pixel = index_fn(img, row+post_rows, col)
            hist_of_col[top_pixel] = hist_of_col[top_pixel] - 1
            hist_of_col[bottom_pixel] = hist_of_col[bottom_pixel] + 1

            hist_of_kernel = hist_of_kernel + hist_of_col

        # 多加一列因为后面会再减一次
        hist_of_kernel = hist_of_kernel + hist_of_cols[0] * pre_cols

        # 更新kernel最右列直方图: `hist_of_right_col`
        # kernel直方图: `hist_of_kernel`
        for col in range(0, cols):
            right_col = col + post_cols
            left_col = col - pre_cols - 1
            top_pixel = index_fn(img, row-pre_rows-1, right_col)
            bottom_pixel = index_fn(img, row+post_rows, right_col)

            hist_of_left_col = hist_of_cols[max(0, left_col + 1)]
            hist_of_right_col = hist_of_cols[min(cols + 1, right_col + 1)]
            hist_of_right_col[top_pixel] = hist_of_right_col[top_pixel] - 1
            hist_of_right_col[bottom_pixel] = hist_of_right_col[bottom_pixel] + 1

            hist_of_kernel = hist_of_kernel + \
                hist_of_right_col - hist_of_left_col
            # 更新`(row, col)`像素
            result[row][col] = getMedianValueFromHistogram(
                hist_of_kernel, middle)

    return result


def getMedianValueFromHistogram(hist, middle):
    pixel = 0
    sum = hist[pixel]
    while pixel < 255 and sum < middle:
        pixel = pixel + 1
        sum = sum + hist[pixel]

    return pixel


def replicaPaddingIndex(img, row, col):
    """
    REPLICA padding 索引方式
    """
    shape = img.shape
    row = max(0, min(row, shape[0]-1))
    col = max(0, min(col, shape[1]-1))
    return img[row, col]


def zeroPaddingIndex(img, row, col):
    """
    ZERO padding 索引方式
    """
    shape = img.shape
    inside = row >= 0 and row < shape[0] and col >= 0 and col < shape[1]
    return inside and img[row, col] or 0


def _kthSmallest(nums, k, l, r):
    if r <= l:
        return nums[k]
    i, j, val = l, r, nums[(l+r)//2]
    while i <= j:
        while i <= j and nums[i] < val:
            i = i + 1
        while i <= j and nums[j] > val:
            j = j - 1
        if i <= j:
            nums[i], nums[j] = nums[j], nums[i]
            i = i + 1
            j = j - 1

    if k <= j:
        return _kthSmallest(nums, k, l, j)
    if k >= i:
        return _kthSmallest(nums, k, i, r)
    return nums[k]


def quickSelectMedianValue(nums):
    r = len(nums) - 1
    return _kthSmallest(nums, r//2, 0, r)


def _quickSort(nums, l, r):
    i, j, val = l, r, nums[(l+r)//2]
    while i <= j:
        while i <= j and nums[i] < val:
            i = i + 1
        while i <= j and nums[j] > val:
            j = j - 1
        if i <= j:
            nums[i], nums[j] = nums[j], nums[i]
            i = i + 1
            j = j - 1
    if l <= j:
        _quickSort(nums, l, j)
    if i <= r:
        _quickSort(nums, i, r)


def quickSort(nums):
    _quickSort(nums, 0, len(nums)-1)


if __name__ == "__main__":
    filepath = len(sys.argv) > 1 and sys.argv[1] or None
    if not filepath:
        proj_dir = os.path.dirname(os.path.abspath(__file__))
        assets_dir = os.path.join(os.path.dirname(proj_dir), 'assets')
        filepath = os.path.join(assets_dir, 'lenna.jpg')

    img = cv2.imread(filepath, 0)
    kernel = np.zeros((5, 5))

    print("img: {}, kernel: {}".format(img.shape, kernel.shape))

    cv2.imshow('lenna', img)

    blured = medianBlur(img, kernel, 'REPLICA')
    cv2.imshow('medianBlur', blured)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
