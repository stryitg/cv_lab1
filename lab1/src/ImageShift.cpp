#include <limits>
#include <iostream>
#include <thread>

#include "ImageShift.hpp"

ImageShift::ImageShift(const Options& options) 
    : _left_image(options.left_image)
    , _right_image(options.right_image)
    , _loss(options.loss)
    , _smoothing_yxyx(GetSmoothing(options.smoothing)) {}
    
ImageShift::Smoothing ImageShift::GetSmoothing(const std::function<float(Shift, Shift)>& 
                                               smoothing_fun) const {
    Smoothing smoothing(kMaxShiftY + 1, std::vector<std::vector<std::vector<float>>>(kMaxShiftX + 1,
                        std::vector<std::vector<float>>(kMaxShiftY + 1, 
                        std::vector<float>(kMaxShiftX + 1))));
    for(uint8_t i = 0; i < kMaxShiftY + 1; ++i) {
        for(uint8_t j = 0; j < kMaxShiftX + 1; ++j) {
            for(uint8_t k = 0; k < kMaxShiftY + 1; ++k) {
                for(uint8_t m = 0; m < kMaxShiftX + 1; ++m) {
                    smoothing[i][j][k][m] = smoothing_fun(Shift{i, j}, Shift{k, m});
                }
            }
        }
    }
    return smoothing;
}
    
cv::Mat ImageShift::GetShift() const {
    ImageShift::Shifts shifts(_left_image.rows, std::vector<Shift>(_left_image.cols));
    std::vector<ImageShift::NodesMap> nodes(_left_image.rows, ImageShift::NodesMap(_left_image.cols, 
                                            Nodes(kMaxShiftY + 1, std::vector<Node>(kMaxShiftX + 1))));
    std::vector<std::thread> threads(std::thread::hardware_concurrency());
    for(size_t count = 0; count < threads.size(); ++count) {
        threads[count] = std::thread([count, &shifts, &nodes, this] () {
            size_t from = count * _left_image.rows / std::thread::hardware_concurrency();
            size_t to = (count + 1) * _left_image.rows / std::thread::hardware_concurrency();
            if(count + 1 == std::thread::hardware_concurrency()) {
                to = _left_image.rows;
            }
            
            for(size_t i = from; i < to; ++i) {
                nodes[i][0] = InitNodes(i);
                for(size_t j = 1; j < _left_image.cols; ++j) {
                    nodes[i][j] = GetNextNodes(nodes[i][j - 1], j, i);
                }
                shifts[i] = GetShifts(nodes[i]);
            }
        });
    }
    
    for(auto& thread: threads) {
        thread.join();
    }
    
    return ToMat(shifts);
}

ImageShift::Nodes ImageShift::InitNodes(size_t row) const {
    Nodes nodes(kMaxShiftY + 1, std::vector<Node>(kMaxShiftX + 1, 
                {.loss = std::numeric_limits<float>::infinity(),
                 .prev = {0, 0}}));
    const size_t count = std::min(row + 1, kMaxShiftY + 1);
    for(size_t i = 0; i < count; ++i) {
        nodes[i][0].loss = _loss(_left_image.at<cv::Vec3b>(row, 0), 
                                 _right_image.at<cv::Vec3b>(row - i, 0));
    }
    return nodes;
}

ImageShift::Nodes ImageShift::GetNextNodes(const ImageShift::Nodes& prev,
                                           size_t index, size_t row) const {
    Nodes nodes(kMaxShiftY + 1, std::vector<Node>(kMaxShiftX + 1, {.loss = std::numeric_limits<float>::infinity(),
                                                                   .prev = {0, 0}}));
    const uint8_t count_x = std::min(index + 1, kMaxShiftX + 1);
    const uint8_t count_y = std::min(row + 1, kMaxShiftY + 1);
    for(size_t k = 0; k < count_y; ++k) {
        const cv::Vec3b* iter = _right_image.ptr<cv::Vec3b>(row - k);
        for(size_t i = 0; i < count_x; ++i) {
            nodes[k][i].loss = _loss(_left_image.at<cv::Vec3b>(row, index), iter[index - i]);
            float min = std::numeric_limits<float>::infinity();
            for(uint8_t m = 0; m < kMaxShiftY + 1; ++m) {
                for(uint8_t j = 0; j < kMaxShiftX + 1; ++j) {
                    const float loss = prev[m][j].loss + _smoothing_yxyx[m][j][k][i]; 
                    if(min > loss) {
                        nodes[k][i].prev = {m, j};
                        min = loss;
                    }
                }
            }
            nodes[k][i].loss += min;
        }
    }
    return nodes;
}

std::vector<ImageShift::Shift> ImageShift::GetShifts(const ImageShift::NodesMap& nodes) const {
    const size_t size = _left_image.cols;
    std::vector<Shift> shifts(size);
    const auto& last = nodes.back();
    Shift shift = {0, 0};
    float min_loss = last[0][0].loss;
    for(uint8_t i = 0; i < last.size(); ++i) {
        for(uint8_t j = 0; j < last[0].size(); ++j) {
            if(min_loss > last[i][j].loss) {
                shift = {i, j};
            }
        }
    }
    // std::cout << nodes.size() << std::endl;
    // std::cout << size << std::endl;
    for(size_t i = 0; i < size; ++i) {
        shifts[size - 1 - i] = shift;
        shift = nodes[size - 1 - i][shift.y][shift.x].prev;
    }
    return shifts;
}

cv::Mat ImageShift::ToMat(const ImageShift::Shifts& shifts) const {
    uint8_t max = 0; 
    cv::Mat mat(shifts.size(), shifts[0].size(), CV_8U);
    for(size_t i = 0; i < shifts.size(); ++i) {
        auto iter = mat.ptr<uint8_t>(i);
        for(size_t j = 0; j < shifts[0].size(); ++j) {
            const auto& shift = shifts[i][j];
            iter[j] = shift.x;
            max = std::max(shift.x, max);
        }
    }
    return mat / max * 256;
}
