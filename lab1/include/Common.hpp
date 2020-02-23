#pragma once 

#include <opencv2/core/core.hpp>
#include <cmath>

enum class LossFunction {
    kL1 = 0,
    kL2,
};

enum class SmoothingFunction {
    kL1 = 0,
    kBetaL1,
};

LossFunction ToLossFunction(const std::string& loss_func) {
    if(loss_func == "L1") {
        return LossFunction::kL1;
    } else if(loss_func == "L2") {
        return LossFunction::kL2;
    } else {
        throw std::runtime_error("Unkown loss function " + loss_func + 
                                 ". Only L1 and L2 are supported");
    }
}

SmoothingFunction ToSmoothingFunction(const std::string& smoothing_func) {
    if(smoothing_func == "L1") {
        return SmoothingFunction::kL1;
    } else if(smoothing_func == "beta-L1") {
        return SmoothingFunction::kBetaL1;
    } else {
        throw std::runtime_error("Unkown smoothing function " + smoothing_func + 
                                 ". Only L1 and beta-L1 are supported");
    }
}

float L1Loss(const cv::Vec3b& vec1, const cv::Vec3b& vec2) {
    float loss = 0.0;
    for(size_t i = 0; i < 3; ++i) {
        loss += std::abs(vec1[i] - vec2[i]);
    }
    return loss;
}

float L2Loss(const cv::Vec3b& vec1, const cv::Vec3b& vec2) {
    float loss = 0.0;
    for(size_t i = 0; i < 3; ++i) {
        loss += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]);
    }
    return std::sqrt(loss); 
}

float L1Smoothing(uint8_t pos1x, uint8_t pos1y,
                  uint8_t pos2x, uint8_t pos2y) {
    return std::abs(static_cast<int64_t>(pos1x - pos2x)) +
           std::abs(static_cast<int64_t>(pos1y - pos2y));
}
