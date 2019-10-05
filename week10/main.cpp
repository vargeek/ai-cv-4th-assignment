#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include <catch2/catch.hpp>
#include "NMS.hpp"


const float kDelta = 0.00001;

TEST_CASE("test the IOU function", "[IOU]") {
    CHECK(fabs(IOU({0,1,0,1}, {1,2,1,2}) - 0) < kDelta);

    CHECK(fabs(IOU({0,1,0,1}, {0.5,1.5,0.5,1.5}) - 1/7.0) < kDelta);

    CHECK(fabs(IOU({0,1,0,1}, {0,1,0,1}) - 1) < kDelta);

    CHECK(fabs(IOU({0,1,0,1}, {0.1,1.1,0,1}) - 0.9/1.1) < kDelta);
}

TEST_CASE("test the NMS function", "[NMS]") {
    // srand(0);

    vector<vector<float>> input = {
        {1, 4, 1, 4, 9},
        {-3+0.1, -1+0.1, -3+0.1, -1+0.1, 7},
        {40, 43, 40, 43, 12},
        {0, 3, 0, 3, 10},
        {10, 12, 10, 12, 6},
        {-2+0.1, 0.1, -2+0.1, 0.1, 8},
        {20, 22, 20, 22, 3},
        {11, 13, 11, 13, 5},
        {41, 42, 41, 42, 11},
        {32-0.1, 34-0.1, 32-0.1, 34-0.1, 1},
        {30, 32, 30, 32, 2},
        {20, 22, 20, 22, 4},
    };
    float threshold = 0.1;

    vector<vector<float>> results = NMS(input, threshold);
    vector<int>got;
    vector<int> want = {12, 10, 8, 6, 4, 2, 1};
    for (size_t i = 0; i < results.size(); i++)
    {
        got.push_back(int(results[i][4]));
    }

    CHECK(got.size() == want.size());
    CHECK(std::equal(got.begin(), got.end(), want.begin()));
}
