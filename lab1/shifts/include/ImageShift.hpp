#pragma once

#include <functional>
#include <vector>
#include <cstdint>

#include <opencv2/core/core.hpp>

class ImageShift {
public:
    struct Shift {
        uint8_t y = 0;
        uint8_t x = 0;
    };
    
    struct Options {
        cv::Mat left_image;
        cv::Mat right_image;
        std::function<float(cv::Vec3b, cv::Vec3b)> loss;
        std::function<float(Shift, Shift)> smoothing;
        int32_t max_shift_x;
        int32_t min_shift_x;
        int32_t max_shift_y;
        int32_t min_shift_y;
    };
    
    ImageShift(const Options& options);
    
    cv::Mat GetShift() const;
    // cv::Mat FilterDense(const cv::Mat& mat, size_t rad) const;
    
private:
    struct Node {
        float loss;
        Shift prev;
    };
    
    using NodesMap = std::vector<std::vector<std::vector<Node>>>;
    using Nodes = std::vector<std::vector<Node>>;
    using Shifts = std::vector<std::vector<Shift>>;
    using Smoothing = std::vector<std::vector<std::vector<std::vector<float>>>>;
    
    Smoothing GetSmoothing(const std::function<float(Shift, Shift)>& smoothing) const;
    
    Nodes InitNodes(size_t row) const;
    Nodes GetNextNodes(const Nodes& prev, size_t index, size_t row) const;
    std::vector<Shift> GetShifts(const NodesMap& nodes) const;
    cv::Mat ToMat(const Shifts& shifts) const;
private:
    static constexpr uint32_t kThreadsCount = 4;
    
    const int32_t kMaxShiftY;
    const int32_t kMinShiftY;
    const size_t kDiffY;
    const int32_t kMaxShiftX;
    const int32_t kMinShiftX;
    const size_t kDiffX;

    cv::Mat _left_image;
    cv::Mat _right_image;
    size_t _init_row;
    size_t _rows;
    size_t _init_col;
    size_t _cols;
    
    std::function<float(cv::Vec3b, cv::Vec3b)> _loss;
    Smoothing _smoothing_yxyx;
};