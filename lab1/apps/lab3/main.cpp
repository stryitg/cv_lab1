#include <vector>
#include <algorithm>
#include <iostream>

#include <boost/program_options.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "Rectification.hpp"
#include "FundamentalMatrix.hpp"

namespace po = boost::program_options;

struct Args {
    std::string left_image_path;
    std::string right_image_path;
    std::string shift_map_path;
};

struct ImagesToRectify {
        cv::Mat& left;
        cv::Mat& right;
        Rectification rect;
};

void DrawOpenCV(const ImagesToRectify& imgs_params);
void DrawOpenCV(const cv::Mat& img, const Eigen::Matrix3d& rect);
cv::Matx33d ToOpenCVMatrix(const Eigen::Matrix3d& m);
void DrawManual(const ImagesToRectify& imgs_params);
void DrawManual(const cv::Mat& img, const Eigen::Matrix3d& rect);
cv::Mat GetRectifiedImg(const cv::Mat& img, const Eigen::Matrix3d& rect);
void ParseCommandLine(int argc, char** argv,
                      FundamentalMatrix::MatchesInternal& internal);

int main(int argc, char** argv) {
    FundamentalMatrix::MatchesInternal internal;
    ParseCommandLine(argc, argv, internal);
    FundamentalMatrix fm(internal);
    const auto f = fm.Get();
        
    ImgShape shp{.height = static_cast<size_t>(internal.left_image.rows),
                 .width = static_cast<size_t>(internal.left_image.cols) };
    const auto matches = fm.GetMatches();
    std::vector<Match> matches_(matches.size());
    std::transform(matches.begin(), matches.end(), matches_.begin(), [] (const auto& v) {
        return Match{.left = v.left, .right = v.right};
    });
    std::cout << "FundamentalMatrix" << std::endl;
    std::cout << f << std::endl;
    const auto rect = GetRectification(f, shp, matches_);
    std::cout << "New fundamental Matrix:" << std::endl;
    std::cout << rect.right.transpose().inverse() * f * rect.left.inverse() << std::endl;
    ImagesToRectify imgs{.left = internal.left_image,
                         .right = internal.right_image,
                         .rect = rect};
    std::cout << "Rectification right:" << std::endl;
    std::cout << rect.right << std::endl;
    std::cout << "Rectification left:" << std::endl;
    std::cout << rect.left << std::endl;
    DrawOpenCV(imgs);
    DrawManual(imgs);
    return 0;
}

void ParseCommandLine(int argc, char** argv, FundamentalMatrix::MatchesInternal& internal) {
    Args args;
    
    po::options_description desc("Options");
    desc.add_options()
        ("left-image", po::value<std::string>(&args.left_image_path)->required(), "path to left image")
        ("right-image", po::value<std::string>(&args.right_image_path)->required(), "path to left image")
        ("shift-map", po::value<std::string>(&args.shift_map_path)->required(), "path to shift map")
        ("min-shift-x", po::value<int32_t>(&internal.min_shift_x)->required(), "min shift in x direction")
        ("min-shift-y", po::value<int32_t>(&internal.min_shift_y)->required(), "min shift in y direction")
        ("help", "produces help message")
    ;
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    
    if(vm.count("help")) {
        std::cout << desc << std::endl;
        std::cout << "Usage:" << std::endl 
        << "./lab1 --left-image limg.jpg --right-image rimg.jpg --shift-map sm.jpg" << std::endl;
        return;
    }
    po::notify(vm);
    
    internal.left_image = cv::imread(args.left_image_path, cv::IMREAD_COLOR);
    if(!internal.left_image.data) {
        throw std::runtime_error("couldn't open " + args.left_image_path);
    }
    internal.right_image = cv::imread(args.right_image_path, cv::IMREAD_COLOR);
    if(!internal.right_image.data) {
        throw std::runtime_error("couldn't open " + args.right_image_path);
    }
    internal.shifts = cv::imread(args.shift_map_path, cv::IMREAD_COLOR);
    if(!internal.shifts.data) {
        throw std::runtime_error("couldn't open " + args.shift_map_path);
    }
}

void DrawOpenCV(const ImagesToRectify& imgs_params) {
    DrawOpenCV(imgs_params.right, imgs_params.rect.right);
    DrawOpenCV(imgs_params.left, imgs_params.rect.left);    
}

void DrawOpenCV(const cv::Mat& img, const Eigen::Matrix3d& rect) {
    cv::Mat img_rect(cv::Size{400, 400}, CV_8UC3);
    const cv::Matx33d rect_ = ToOpenCVMatrix(rect);
    cv::warpPerspective(img, img_rect, rect_, cv::Size{400, 400});
    cv::namedWindow("OPENCV RECT", cv::WINDOW_AUTOSIZE);
    cv::imshow("OPENCV RECT", img_rect);
    cv::waitKey(0);
}

cv::Matx33d ToOpenCVMatrix(const Eigen::Matrix3d& m) {
    const auto m_ = m / m.norm();
    cv::Matx33d out(m_(0, 0), m_(1, 0), m_(0, 2),
                    m_(1, 0), m_(1, 1), m_(1, 2),
                    m_(2, 0), m_(2, 1), m_(2, 2));
    return out;
}

void DrawManual(const ImagesToRectify& imgs_params) {
    DrawManual(imgs_params.right, imgs_params.rect.right);
    DrawManual(imgs_params.left, imgs_params.rect.left);
}

void DrawManual(const cv::Mat& img, const Eigen::Matrix3d& rect) {
    const auto img_rect = GetRectifiedImg(img, rect);
    cv::namedWindow("MANUAL RECT", cv::WINDOW_AUTOSIZE);
    cv::imshow("MANUAL RECT", img_rect);
    cv::waitKey(0);
}

cv::Mat GetRectifiedImg(const cv::Mat& img, const Eigen::Matrix3d& rect) {
    const auto rect_inv = rect.inverse(); 
    cv::Mat img_rect(cv::Size{400, 400}, CV_8UC3);
    img_rect = cv::Scalar(0, 0, 0);
    for(size_t i = 0; i < img.rows; ++i) {
        for(size_t j = 0; j < img.cols; ++j) {
            Eigen::Vector3d v;
            v << (double) j, (double) i, 1.0;
            const auto new_pos = ProjectiveMult(rect_inv, v);
            if (new_pos(0) >= 0.0 && new_pos(0) <= img.cols &&
                new_pos(1) >= 0.0 && new_pos(1) <= img.rows) {
                img_rect.at<cv::Vec3b>(i, j) = img.at<cv::Vec3b>(new_pos(1),
                                                                 new_pos(0));
            }
        }
    } 
    return img_rect;
}
