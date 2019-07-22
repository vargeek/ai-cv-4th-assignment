#include "c1_median_blur.hpp"
#include <algorithm>

static u8 replicaPaddingIndex(vector<vector<u8>> &img, int row, int col);
static u8 zeroPaddingIndex(vector<vector<u8>> &img, int row, int col);

vector<vector<u8>> medianBlur(vector<vector<u8>> &img, vector<vector<u8>> kernel, string padding_way) {
    std::transform(padding_way.begin(), padding_way.end(), padding_way.begin(), ::toupper);

    u8 (*index_fn)(vector<vector<u8>> &, int, int) = padding_way == "REPLICA" ? replicaPaddingIndex : zeroPaddingIndex;

    int rows = img.size();
    int cols = rows > 0 ? img[0].size() : 0;
    int krows = kernel.size();
    int kcols = krows > 0 ? kernel[0].size() : 0;
    int ksize = krows * kcols;

    int row_offset = -(krows / 2);
    int col_offset = -(kcols / 2);

    vector<u8> pixels = vector<u8>(ksize);
    vector<vector<u8>> result = vector<vector<u8>>(rows);
    for (int row = 0; row < rows; row++) {
        result[row] = vector<u8>(cols);
    }

    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            int row_start = row + row_offset;
            int col_start = col + col_offset;
            int idx = 0;
            for (int r = row_start; r < row_start + krows; r++) {
                for (int c = col_start; c < col_start + kcols; c++) {
                    pixels[idx++] = index_fn(img, r, c);
                }
            }
            sort(pixels.begin(), pixels.end());
            result[row][col] = pixels[(ksize - 1) / 2];
        }
    }

    return result;
}

vector<vector<u8>> medianBlurQuickSort(vector<vector<u8>> &img, vector<vector<u8>> kernel, string padding_way) {
    std::transform(padding_way.begin(), padding_way.end(), padding_way.begin(), ::toupper);

    u8 (*index_fn)(vector<vector<u8>> &, int, int) = padding_way == "REPLICA" ? replicaPaddingIndex : zeroPaddingIndex;

    int rows = img.size();
    int cols = rows > 0 ? img[0].size() : 0;
    int krows = kernel.size();
    int kcols = krows > 0 ? kernel[0].size() : 0;
    int ksize = krows * kcols;

    int row_offset = -(krows / 2);
    int col_offset = -(kcols / 2);

    vector<u8> pixels = vector<u8>(ksize);
    vector<vector<u8>> result = vector<vector<u8>>(rows);
    for (int row = 0; row < rows; row++) {
        result[row] = vector<u8>(cols);
    }

    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            int row_start = row + row_offset;
            int col_start = col + col_offset;
            int idx = 0;
            for (int r = row_start; r < row_start + krows; r++) {
                for (int c = col_start; c < col_start + kcols; c++) {
                    pixels[idx++] = index_fn(img, r, c);
                }
            }
            quickSort(pixels);
            result[row][col] = pixels[(ksize - 1) / 2];
        }
    }

    return result;
}

vector<vector<u8>> medianBlurQuickSelect(vector<vector<u8>> &img, vector<vector<u8>> kernel, string padding_way) {
    std::transform(padding_way.begin(), padding_way.end(), padding_way.begin(), ::toupper);

    u8 (*index_fn)(vector<vector<u8>> &, int, int) = padding_way == "REPLICA" ? replicaPaddingIndex : zeroPaddingIndex;

    int rows = img.size();
    int cols = rows > 0 ? img[0].size() : 0;
    int krows = kernel.size();
    int kcols = krows > 0 ? kernel[0].size() : 0;
    int ksize = krows * kcols;

    int row_offset = -(krows / 2);
    int col_offset = -(kcols / 2);

    vector<u8> pixels = vector<u8>(ksize);
    vector<vector<u8>> result = vector<vector<u8>>(rows);
    for (int row = 0; row < rows; row++) {
        result[row] = vector<u8>(cols);
    }

    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            int row_start = row + row_offset;
            int col_start = col + col_offset;
            int idx = 0;
            for (int r = row_start; r < row_start + krows; r++) {
                for (int c = col_start; c < col_start + kcols; c++) {
                    pixels[idx++] = index_fn(img, r, c);
                }
            }
            result[row][col] = quickSelectMedianValue(pixels);
        }
    }

    return result;
}

static const int kBins = 256;

static void add_hist(vector<int> &hist1, vector<int> &hist2) {
    for (int i = 0; i < kBins; i++) {
        hist1[i] += hist2[i];
    }
}

static void add_hist(vector<int> &hist1, vector<int> &hist2, int mul) {
    for (int i = 0; i < kBins; i++) {
        hist1[i] += hist2[i] * mul;
    }
}

static void shift_hist(vector<int> &hist1, vector<int> &hist2, vector<int> &hist3) {
    for (int i = 0; i < kBins; i++) {
        hist1[i] += hist2[i] - hist3[i];
    }
}

static u8 getMedianValueFromHistogram(vector<int> &hist, int middle) {
    u8 pixel = 0;
    int sum = hist[pixel];
    while (pixel < kBins - 1 && sum < middle) {
        pixel++;
        sum += hist[pixel];
    }
    return pixel;
}

vector<vector<u8>> medianBlurHistogram(vector<vector<u8>> &img, vector<vector<u8>> kernel, string padding_way) {
    std::transform(padding_way.begin(), padding_way.end(), padding_way.begin(), ::toupper);

    u8 (*index_fn)(vector<vector<u8>> &, int, int) = padding_way == "REPLICA" ? replicaPaddingIndex : zeroPaddingIndex;

    // 图片行数、列数
    int rows = img.size();
    int cols = rows > 0 ? img[0].size() : 0;
    // kernel行数、列数
    int krows = kernel.size();
    int kcols = krows > 0 ? kernel[0].size() : 0;
    // kernel在当前点上面的行数
    int pre_rows = krows / 2;
    // kernel在当前点下面的行数
    int post_rows = (krows - 1) / 2;
    // kernel在当前点左边的列数
    int pre_cols = kcols / 2;
    // kernel当前点右边的列数
    int post_cols = (kcols - 1) / 2;
    int middle = (krows * kcols + 1) / 2;

    vector<vector<u8>> result(rows);
    for (int row = 0; row < rows; row++) {
        result[row] = vector<u8>(cols);
    }

    // 每列的直方图, 256 bin
    vector<vector<int>> hist_of_cols(cols + 2);
    for (int col = 0; col < cols + 2; col++) {
        hist_of_cols[col] = vector<int>(kBins);
    }

    // 当前kernel直方图, 256 bin
    vector<int> hist_of_kernel(kBins);

    // 初始化每列直方图
    for (int col = -1; col < cols + 1; col++) {
        u8 pixel = index_fn(img, -1, col);
        // 多加一是因为更新`hist_of_kernel`会再减一次
        vector<int> &hist_of_col = hist_of_cols[1 + col];
        hist_of_col[pixel] += pre_rows + 1;
        for (int row = 0; row < post_rows; row++) {
            pixel = index_fn(img, row, col);
            hist_of_col[pixel]++;
        }
    }

    // 过滤后的图像
    for (int row = 0; row < rows; row++) {
        // 初始化当前kernel直方图
        std::fill(hist_of_kernel.begin(), hist_of_kernel.end(), 0);

        // 更新前 `post_cols+1` 列直方图
        for (int col = -1; col < post_cols; col++) {
            vector<int> &hist_of_col = hist_of_cols[col + 1];
            u8 top_pixel = index_fn(img, row - pre_rows - 1, col);
            u8 bottom_pixel = index_fn(img, row + post_rows, col);
            hist_of_col[top_pixel]--;
            hist_of_col[bottom_pixel]++;
            add_hist(hist_of_kernel, hist_of_col);
        }

        // 多加一列因为后面会再减一次
        add_hist(hist_of_kernel, hist_of_cols[0], pre_cols);

        // 更新kernel最右列直方图: `hist_of_right_col`
        // kernel直方图: `hist_of_kernel`
        for (int col = 0; col < cols; col++) {
            int right_col = col + post_cols;
            int left_col = col - pre_cols - 1;
            u8 top_pixel = index_fn(img, row - pre_rows - 1, right_col);
            u8 bottom_pixel = index_fn(img, row + post_rows, right_col);

            vector<int> &hist_of_left_col = hist_of_cols[max(0, left_col + 1)];
            vector<int> &hist_of_right_col = hist_of_cols[min(cols + 1, right_col + 1)];
            hist_of_right_col[top_pixel]--;
            hist_of_right_col[bottom_pixel]++;

            shift_hist(hist_of_kernel, hist_of_right_col, hist_of_left_col);

            // 更新`(row, col)`像素
            result[row][col] = getMedianValueFromHistogram(hist_of_kernel, middle);
        }
    }

    return result;
}

static u8 replicaPaddingIndex(vector<vector<u8>> &img, int row, int col) {
    int rows = img.size();
    int cols = rows > 0 ? img[0].size() : 0;
    row = max<int>(0, min<int>(row, rows - 1));
    col = max<int>(0, min<int>(col, cols - 1));
    return img[row][col];
}

static u8 zeroPaddingIndex(vector<vector<u8>> &img, int row, int col) {
    int rows = img.size();
    int cols = rows > 0 ? img[0].size() : 0;
    bool inside = row >= 0 && row < rows && col >= 0 && col < cols;

    return inside ? img[row][col] : 0;
}

static void _quickSort(vector<u8> &nums, int l, int r) {
    int i = l, j = r, val = nums[(l + r) / 2];
    while (i <= j) {
        while (i <= j && nums[i] < val) {
            i++;
        }
        while (i <= j && nums[j] > val) {
            j--;
        }
        if (i <= j) {
            u8 tmp = nums[i];
            nums[i] = nums[j];
            nums[j] = tmp;
            i++;
            j--;
        }
    }
    if (l <= j) {
        _quickSort(nums, l, j);
    }
    if (i <= r) {
        _quickSort(nums, i, r);
    }
}
void quickSort(vector<u8> &nums) {
    _quickSort(nums, 0, nums.size() - 1);
}

static u8 _kthSmallest(vector<u8> &nums, int k, int l, int r) {
    if (r <= l) {
        return nums[k];
    }

    int i = l, j = r, val = nums[(l + r) / 2];
    while (i <= j) {
        while (i <= j && nums[i] < val) {
            i++;
        }
        while (i <= j && nums[j] > val) {
            j--;
        }
        if (i <= j) {
            u8 tmp = nums[i];
            nums[i] = nums[j];
            nums[j] = tmp;
            i++;
            j--;
        }
    }
    if (k <= j) {
        return _kthSmallest(nums, k, l, j);
    }
    if (k >= i) {
        return _kthSmallest(nums, k, i, r);
    }
    return nums[k];
}

u8 quickSelectMedianValue(vector<u8> &nums) {
    int r = nums.size() - 1;
    return _kthSmallest(nums, r / 2, 0, r);
}
