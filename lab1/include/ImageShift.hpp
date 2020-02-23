#pragma once

#include <functional>
#include <vector>
#include <cstdint>

#include <opencv2/core/core.hpp>

class ImageShift {
public:
    struct Shift {
        uint8_t y;
        uint8_t x;
    };
    
    struct Options {
        cv::Mat left_image;
        cv::Mat right_image;
        std::function<float(cv::Vec3b, cv::Vec3b)> loss;
        std::function<float(Shift, Shift)> smoothing;
    };
    
    ImageShift(const Options& options);
    
    cv::Mat GetShift() const;
    
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
    static constexpr size_t kMaxShiftX = 50;
    static constexpr size_t kMaxShiftY = 10;
    
    // std::vector<>
    
    cv::Mat _left_image;
    cv::Mat _right_image;
    
    std::function<float(cv::Vec3b, cv::Vec3b)> _loss;
    
    Smoothing _smoothing_yxyx;
};