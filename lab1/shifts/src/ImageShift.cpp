#include <limits>
#include <iostream>
#include <thread>

#include "ImageShift.hpp"

ImageShift::ImageShift(const Options& options) 
    : kMaxShiftY(options.max_shift_y)
    , kMinShiftY(options.min_shift_y)
    , kDiffY((size_t) kMaxShiftY - kMinShiftY)
    , kMaxShiftX(options.max_shift_x)
    , kMinShiftX(options.min_shift_x)
    , kDiffX((size_t) kMaxShiftX - kMinShiftX)
    , _left_image(options.left_image)
    , _right_image(options.right_image)
    , _init_row(kMinShiftY > 0 ? kMinShiftY : 0)
    , _rows(_left_image.rows - std::abs(kMinShiftY))
    , _init_col(kMinShiftX > 0 ? kMinShiftX : 0)
    , _cols(_left_image.cols - std::abs(kMinShiftX))
    , _loss(options.loss)
    , _smoothing_yxyx(GetSmoothing(options.smoothing)) {
    if(_left_image.rows != _right_image.rows ||
       _left_image.cols != _right_image.cols ||
       _left_image.rows == 0 || _left_image.cols == 0) {
        throw std::logic_error("Sizes");   
    }
    if(kMaxShiftX < kMinShiftX) {
        throw std::logic_error("max_shift_x < min_shift_x");
    }
    if(kMaxShiftY < kMinShiftY) {
        throw std::logic_error("max_shift_y < min_shift_y");
    }
}
    
ImageShift::Smoothing ImageShift::GetSmoothing(const std::function<float(Shift, Shift)>& 
                                               smoothing_fun) const {
    Smoothing smoothing(kDiffY + 1, std::vector<std::vector<std::vector<float>>>(kDiffX + 1,
                        std::vector<std::vector<float>>(kDiffY + 1, 
                        std::vector<float>(kDiffX + 1))));
    for(uint8_t i = 0; i < kDiffY + 1; ++i) {
        for(uint8_t j = 0; j < kDiffX + 1; ++j) {
            for(uint8_t k = 0; k < kDiffY + 1; ++k) {
                for(uint8_t m = 0; m < kDiffX + 1; ++m) {
                    smoothing[i][j][k][m] = smoothing_fun(Shift{i, j}, Shift{k, m});
                }
            }
        }
    }
    return smoothing;
}
    
cv::Mat ImageShift::GetShift() const {
    ImageShift::Shifts shifts(_left_image.rows, std::vector<Shift>(_left_image.cols));
    std::vector<ImageShift::NodesMap> nodes(_rows, ImageShift::NodesMap(_cols, 
                                            Nodes(kDiffY + 1, 
                                            std::vector<Node>(kDiffX  + 1))));
    std::vector<std::thread> threads(kThreadsCount);
    for(size_t count = 0; count < threads.size(); ++count) {
        threads[count] = std::thread([count, &shifts, &nodes, this] () {
            size_t from = count * _rows / kThreadsCount;
            size_t to = (count + 1) * _rows / kThreadsCount;
            if(count + 1 == kThreadsCount) {
                to = _rows;
            }
            
            for(size_t i = from; i < to; ++i) {
                nodes[i][0] = InitNodes(i);
                for(size_t j = 1; j < _cols; ++j) {
                    nodes[i][j] = GetNextNodes(nodes[i][j - 1], j, i);
                }
                shifts[i + _init_row] = GetShifts(nodes[i]);
            }
        });
    }
    
    for(auto& thread: threads) {
        thread.join();
    }
    
    return ToMat(shifts);
}

ImageShift::Nodes ImageShift::InitNodes(size_t row) const {
    row += _init_row;
    Nodes nodes(kDiffY + 1, std::vector<Node>(kDiffX + 1,
                {.loss = std::numeric_limits<float>::infinity(),
                 .prev = {0, 0}}));
    size_t count_y = (size_t) (std::min((int32_t) row + 1, kMaxShiftY + 1) - kMinShiftY);
    size_t count_x = (size_t) (std::min((int32_t) _init_col + 1, kMaxShiftX + 1) - kMinShiftX);
    for(size_t i = 0; i < count_y; ++i) {;        
        for(size_t j = 0; j < count_x; ++j) {
            nodes[i][j].loss = _loss(_left_image.at<cv::Vec3b>(row, j), 
                                     _right_image.at<cv::Vec3b>(row - i - kMinShiftY, j));
        }
    }
    return nodes;
}

ImageShift::Nodes ImageShift::GetNextNodes(const ImageShift::Nodes& prev,
                                           size_t col, size_t row) const {
    col += _init_col; 
    row += _init_row;
    Nodes nodes(kDiffY + 1, std::vector<Node>(kDiffX + 1, {.loss = std::numeric_limits<float>::infinity(),
                                                           .prev = {0, 0}}));
    size_t count_x = (size_t) (std::min((int32_t) col + 1, kMaxShiftX + 1) - kMinShiftX);
    size_t count_y = (size_t) (std::min((int32_t) row + 1, kMaxShiftY + 1) - kMinShiftY);
    for(size_t k = 0; k < count_y; ++k) {
        const cv::Vec3b* iter = _right_image.ptr<cv::Vec3b>(row - k - kMinShiftY);
        for(size_t i = 0; i < count_x; ++i) {
            nodes[k][i].loss = _loss(_left_image.at<cv::Vec3b>(row, col), iter[col - i - kMinShiftX]);
            float min = std::numeric_limits<float>::infinity();
            for(uint8_t m = 0; m < kDiffY + 1; ++m) {
                for(uint8_t j = 0; j < kDiffX + 1; ++j) {
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
    const size_t size = _cols;
    std::vector<Shift> shifts(_left_image.cols);
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

    for(size_t i = 0; i < size; ++i) {
        shifts[_init_col + size - 1 - i] = shift;
        shift = nodes[size - 1 - i][shift.y][shift.x].prev;
    }
    return shifts;
}

cv::Mat ImageShift::ToMat(const ImageShift::Shifts& shifts) const {
    cv::Mat mat(shifts.size(), shifts[0].size(), CV_8UC3);
    mat = cv::Scalar(0, 0, 0);
    for(size_t i = 0; i < shifts.size(); ++i) {
        auto iter = mat.ptr<cv::Vec3b>(i);
        for(size_t j = 0; j < shifts[0].size(); ++j) {
            const auto& shift = shifts[i][j];
            iter[j][0] = shift.y;
            iter[j][1] = shift.x;
        }
    }
    return mat;
}

