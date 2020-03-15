#include <iostream>

#include <boost/program_options.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "ImageShift.hpp"
#include "FundamentalMatrix.hpp"
#include "Common.hpp"

namespace po = boost::program_options;

struct Args {
    std::string left_image_path;
    std::string right_image_path;
    std::string shift_map_path;
};

struct Line {
    cv::Point begin;
    cv::Point end;
};

void ParseCommandLine(int argc, char** argv, FundamentalMatrix::MatchesInternal& internal);
std::vector<cv::Point> GetRandomPoints(size_t count, int max_y, int max_x);
void Display(cv::Mat left_image, cv::Mat right_image,
             const Eigen::Matrix3d& fm, const std::vector<cv::Point>& pts);
Line GetLine(const cv::Point& pt, const Eigen::Vector3d& epipole);

int main(int argc, char** argv) {
    try {
        FundamentalMatrix::MatchesInternal internal;
        ParseCommandLine(argc, argv, internal);
        FundamentalMatrix fm(internal);
        const auto f = fm.Get();
        std::cout << f << std::endl;
        const auto pts = GetRandomPoints(10, internal.left_image.cols, 
                                         internal.left_image.rows);
        Display(internal.left_image, internal.right_image, f, pts);
    } catch (const std::exception& ex) {
        std::cout << ex.what() << std::endl;
        return 1;
    }
    
    
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

std::vector<cv::Point> GetRandomPoints(size_t count, int max_y, int max_x) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis_y(0, max_y);
    std::uniform_int_distribution<int> dis_x(0, max_x);
    
    std::vector<cv::Point> pts(count);
    for(auto& pt: pts) {
        pt = {dis_x(gen), dis_y(gen)};
    }
    return pts;
}

void Display(cv::Mat left_image, cv::Mat right_image,
             const Eigen::Matrix3d& fm, const std::vector<cv::Point>& pts) {
    const int thickness = 2;
    const int line_type = 8;
    const auto epipole = FundamentalMatrix::GetEpipole(fm);
    for(const auto& pt: pts) {
        const auto line_ = GetLine(pt, epipole);
        std::cout << line_.begin << std::endl;
        std::cout << line_.end << std::endl;
        line(left_image, line_.begin, line_.end,
             cv::Scalar(0, 0, 0), thickness, line_type);
        Eigen::Vector3d v;
        v << pt.x, pt.y, 1.0;
        const auto r_line = fm * v;
        std::cout << (( r_line[2] + 1000 * r_line[0]) / r_line[1]) << std::endl;
        const cv::Point begin{2000, - (int) (( r_line[2] + 2000 * r_line[0]) / r_line[1])}; 
        const cv::Point end{0, - (int) ((r_line[2]) / r_line[1])};
        line(right_image, begin, end,
             cv::Scalar(0, 0, 0), thickness, line_type);
    }
    cv::imshow("left", left_image);
    cv::imshow("right", right_image);
    cv::waitKey(0);
}

Line GetLine(const cv::Point& pt, const Eigen::Vector3d& epipole) {
    const auto diff_x = epipole(0) - pt.x;
    if(std::abs(diff_x) < 0.001) {
        return {.begin = {pt.x, 0},
                .end = {pt.x, 2000}};
    }
    
    const auto k = (double) (epipole(1) - pt.y) / diff_x; 
    std::cout << k << std::endl;
    return {.begin = {0, (int) ((0 - pt.x) * k + pt.y)},
            .end = {2000, (int) ((2000 - pt.x) * k + pt.y)}};
}
