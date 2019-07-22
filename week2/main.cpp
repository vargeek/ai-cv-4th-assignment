#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include <time.h>
#include <algorithm>
#include <catch2/catch.hpp>
#include "c1_median_blur.hpp"

vector<u8> random_vec(size_t len) {
    vector<u8> vec = vector<u8>(len);
    for (size_t i = 0; i < len; i++) {
        vec[i] = rand() % 256;
    }
    return vec;
}

vector<u8> copy_vec(const vector<u8> &src) {
    vector<u8> dst = vector<u8>(src.size());
    for (size_t i = 0; i < src.size(); i++) {
        dst[i] = src[i];
    }
    return dst;
}

vector<vector<u8>> random_img(size_t rows, size_t cols) {
    vector<vector<u8>> img = vector<vector<u8>>(rows);
    for (int row = 0; row < rows; row++) {
        img[row] = random_vec(cols);
    }
    return img;
}

vector<vector<u8>> copy_img(const vector<vector<u8>> src) {
    vector<vector<u8>> dst = vector<vector<u8>>(src.size());
    for (int row = 0; row < src.size(); row++) {
        dst[row] = copy_vec(src[row]);
    }
    return dst;
}

bool check_img(const vector<vector<u8>> &img1, const vector<vector<u8>> &img2) {
    int rows = img1.size();
    if (rows != img2.size()) {
        return false;
    }
    int cols = rows > 0 ? img1[0].size() : 0;
    if (rows > 0 && cols != img2[0].size()) {
        return false;
    }
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            if (img1[row][col] != img2[row][col]) {
                printf("row: %d, col: %d, %d, %d\n", row, col, img1[row][col], img2[row][col]);
                return false;
            }
        }
    }
    return true;
}

void print_img(const vector<vector<u8>> &img1) {
    int rows = img1.size();
    int cols = rows > 0 ? img1[0].size() : 0;
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            printf("\t%d", img1[row][col]);
        }
        printf("\n");
    }
}

TEST_CASE("test the quickSort function", "[quickSort]") {
    // srand(0);
    srand((unsigned)time(NULL));

    vector<u8> nums = random_vec(1000);
    vector<u8> copied = copy_vec(nums);
    quickSort(nums);
    sort(copied.begin(), copied.end());

    CHECK(std::equal(nums.begin(), nums.end(), copied.begin()));
}

TEST_CASE("test the quickSelectMedianValue function", "[quickSelectMedianValue]") {
    srand((unsigned)time(NULL));

    vector<u8> nums = random_vec(1000);
    vector<u8> copied = copy_vec(nums);
    sort(copied.begin(), copied.end());

    CHECK(quickSelectMedianValue(nums) == copied[(copied.size() - 1) / 2]);
}

TEST_CASE("benchmark", "[benchmark]") {
    srand((unsigned)time(NULL));
    vector<u8> nums = random_vec(1000);

    BENCHMARK_ADVANCED("quickSelectMedianValue")
    (Catch::Benchmark::Chronometer meter) {
        vector<u8> copied = copy_vec(nums);
        meter.measure([&copied] {
            quickSelectMedianValue(copied);
        });
    };

    BENCHMARK_ADVANCED("quickSort")
    (Catch::Benchmark::Chronometer meter) {
        vector<u8> copied = copy_vec(nums);
        meter.measure([&copied] {
            quickSort(copied);
        });
    };
    BENCHMARK_ADVANCED("sort")
    (Catch::Benchmark::Chronometer meter) {
        vector<u8> copied = copy_vec(nums);
        meter.measure([&copied] {
            sort(copied.begin(), copied.end());
        });
    };
}

TEST_CASE("test the medianBlur function", "[medianBlur]") {
    // srand(0);
    srand((unsigned)time(NULL));

    vector<vector<u8>> img = random_img(256, 256);
    vector<vector<u8>> kernel = random_img(4, 4);

    vector<vector<u8>> result1 = medianBlur(img, kernel, "REPLICA");
    vector<vector<u8>> result2 = medianBlurQuickSelect(img, kernel, "REPLICA");

    CHECK(check_img(result1, result2));
}

TEST_CASE("test the medianBlurHistogram function", "[medianBlurHistogram]") {
    // srand(0);
    srand((unsigned)time(NULL));

    vector<vector<u8>> img = random_img(256, 256);
    vector<vector<u8>> kernel = random_img(4, 4);

    vector<vector<u8>> result1 = medianBlurQuickSelect(img, kernel, "REPLICA");
    vector<vector<u8>> result2 = medianBlurHistogram(img, kernel, "REPLICA");

    CHECK(check_img(result1, result2));
}

TEST_CASE("medianBlur 256x256x4x4benchmark", "[benchmark]") {
    srand((unsigned)time(NULL));

    vector<vector<u8>> img = random_img(256, 256);
    vector<vector<u8>> kernel = random_img(4, 4);

    BENCHMARK_ADVANCED("medianBlur")
    (Catch::Benchmark::Chronometer meter) {
        meter.measure([&img, &kernel] {
            medianBlur(img, kernel, "REPLICA");
        });
    };

    BENCHMARK_ADVANCED("medianBlurQuickSort")
    (Catch::Benchmark::Chronometer meter) {
        meter.measure([&img, &kernel] {
            medianBlurQuickSort(img, kernel, "REPLICA");
        });
    };

    BENCHMARK_ADVANCED("medianBlurQuickSelect")
    (Catch::Benchmark::Chronometer meter) {
        meter.measure([&img, &kernel] {
            medianBlurQuickSelect(img, kernel, "REPLICA");
        });
    };

    BENCHMARK_ADVANCED("medianBlurHistogram")
    (Catch::Benchmark::Chronometer meter) {
        meter.measure([&img, &kernel] {
            medianBlurHistogram(img, kernel, "REPLICA");
        });
    };
}

TEST_CASE("medianBlur 100x100x20x20 benchmark", "[benchmark]") {
    srand((unsigned)time(NULL));

    vector<vector<u8>> img = random_img(100, 100);
    vector<vector<u8>> kernel = random_img(20, 20);

    BENCHMARK_ADVANCED("medianBlur")
    (Catch::Benchmark::Chronometer meter) {
        meter.measure([&img, &kernel] {
            medianBlur(img, kernel, "REPLICA");
        });
    };

    BENCHMARK_ADVANCED("medianBlurQuickSort")
    (Catch::Benchmark::Chronometer meter) {
        meter.measure([&img, &kernel] {
            medianBlurQuickSort(img, kernel, "REPLICA");
        });
    };

    BENCHMARK_ADVANCED("medianBlurQuickSelect")
    (Catch::Benchmark::Chronometer meter) {
        meter.measure([&img, &kernel] {
            medianBlurQuickSelect(img, kernel, "REPLICA");
        });
    };

    BENCHMARK_ADVANCED("medianBlurHistogram")
    (Catch::Benchmark::Chronometer meter) {
        meter.measure([&img, &kernel] {
            medianBlurHistogram(img, kernel, "REPLICA");
        });
    };
}