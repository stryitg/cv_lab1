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

void ParseCommandLine(int argc, char** argv, FundamentalMatrix::MatchesInternal& internal);
void Display(cv::Mat left_image, cv::Mat right_image,
             const Eigen::Matrix3d& fm, const std::vector<cv::Point>& pts);
             
int main(int argc, char** argv) {
    try {
        FundamentalMatrix::MatchesInternal internal;
        ParseCommandLine(argc, argv, internal);
        FundamentalMatrix fm(internal);
        const auto f = fm.Get();
        std::cout << f << std::endl;
        Display(internal.left_image, internal.right_image, f, {{400, 0}, {400, 100},
        {400, 200}, {400, 300}, {400, 400}});
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

void Display(cv::Mat left_image, cv::Mat right_image,
             const Eigen::Matrix3d& fm, const std::vector<cv::Point>& pts) {
    const int thickness = 2;
    const int line_type = 8;
    const auto epipole = FundamentalMatrix::GetEpipole(fm);
    for(size_t i = 0; i < pts.size(); ++i) {
        line(left_image, {(int) -epipole(0), (int) -epipole(1)}, pts[i],
             cv::Scalar(0, 0, 0), thickness, line_type);
        Eigen::Vector3d v;
        v << pts[i].x, pts[i].y, 1.0;
        const auto r_line = fm * v;
        const cv::Point begin{1000, - (int) (( r_line[2] + 1000 * r_line[0]) / r_line[1])}; 
        const cv::Point end{-1000, - (int) ((r_line[2] - 1000 * r_line[0]) / r_line[1])};
        line(right_image, begin, end,
             cv::Scalar(0, 0, 0), thickness, line_type);
    }
    cv::imshow("left", left_image);
    cv::imshow("right", right_image);
    cv::waitKey(0);
}
