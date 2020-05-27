#include <iostream>
#include <string>

#include <boost/program_options.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

#include "ImageShift.hpp"
#include "Common.hpp"

namespace po = boost::program_options;

struct Args {
    std::string left_image_path;
    std::string right_image_path;
    std::string loss_func;
    std::string smoothing_func;
    float alpha_smoothing_param;
    float beta_smoothing_param;
};

void ParseCommandLine(int argc, char** argv, ImageShift::Options& options, std::string& out);

int main(int argc, char** argv) {
    try {
        ImageShift::Options options;
        std::string out_path;
        ParseCommandLine(argc, argv, options, out_path);
        ImageShift image_shift(options);
        const auto shift_map = image_shift.GetShift();
        cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
        cv::imshow("Display window", shift_map);
        cv::waitKey(0);
        if(!out_path.empty()) {
            std::vector<int> compression_params;
            compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
            compression_params.push_back(0);
            cv::imwrite(out_path, shift_map, compression_params);
        }
    } catch (const std::exception& ex) {
        std::cout << ex.what() << std::endl;
        return 1;
    }
    
    
    return 0;
}

void ParseCommandLine(int argc, char** argv, ImageShift::Options& options, std::string& out) {
    Args args;
    
    po::options_description desc("Options");
    desc.add_options()
        ("left-image", po::value<std::string>(&args.left_image_path)->required(), "path to left image")
        ("right-image", po::value<std::string>(&args.right_image_path)->required(), "path to left image")
        ("loss-func", po::value<std::string>(&args.loss_func)->required(), "loss function type (L1, L2)")
        ("smoothing-func", po::value<std::string>(&args.smoothing_func)->required(), "loss function type (L1, beta-L1)")
        ("alpha-smoothing-param", po::value<float>(&args.alpha_smoothing_param)->required(), "alpha smothing parameter")
        ("beta-smoothing-param", po::value<float>()->default_value(1.0), "beta smoothing param for beta-L1 smoothing function")
        ("max-shift-x", po::value<int32_t>(&options.max_shift_x)->required(), "max shift in x direction")
        ("min-shift-x", po::value<int32_t>(&options.min_shift_x)->default_value(0), "min shift in x direction")
        ("max-shift-y", po::value<int32_t>(&options.max_shift_y)->required(), "max shift in y direction")
        ("min-shift-y", po::value<int32_t>(&options.min_shift_y)->default_value(0), "min shift in y direction")
        ("out", po::value<std::string>(&out)->default_value(""), "path to save output image; don't provide for not saving")
        ("help", "produces help message")
    ;
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    
    if(vm.count("help")) {
        std::cout << desc << std::endl;
        std::cout << "Usage:" << std::endl 
        << "./lab1 --left-image limg.jpg --right-image rimg.jpg --loss-func L2"
        << " --smoothing-func L1 --alpha_smoothing_param 0.3" << std::endl;
        return;
    }
    po::notify(vm);
    
    const auto left_image = cv::imread(args.left_image_path, cv::IMREAD_COLOR);
    if(!left_image.data) {
        throw std::runtime_error("couldn't open " + args.left_image_path);
    }
    if(false && left_image.rows > 400) {
        cv::resize(left_image, options.left_image, cv::Size(), 400.0 / left_image.rows, 400.0 / left_image.rows);
    } else {
        options.left_image = std::move(left_image);
    }
    
    const auto right_image = cv::imread(args.right_image_path, cv::IMREAD_COLOR);
    if(!right_image.data) {
        throw std::runtime_error("couldn't open " + args.right_image_path);
    }
    if(false && right_image.rows > 400) {
        cv::resize(right_image, options.right_image, cv::Size(), 400.0 / right_image.rows, 400.0 / right_image.rows);
    } else {
        options.right_image = std::move(right_image);
    }
    
    const auto loss_func = ToLossFunction(args.loss_func);
    const auto smoothing_func = ToSmoothingFunction(args.smoothing_func);
    if(smoothing_func == SmoothingFunction::kBetaL1) {
        args.beta_smoothing_param = vm["beta-smoothing-param"].as<float>();
    }
    // options.left_image.resize(400);
    
    if(loss_func == LossFunction::kL1) {
        options.loss = L1Loss;
    } else if(loss_func == LossFunction::kL2) {
        options.loss = L2Loss;
    }
    
    if(smoothing_func == SmoothingFunction::kL1) {
        options.smoothing = [alpha = args.alpha_smoothing_param] (ImageShift::Shift pos1,
                                                                  ImageShift::Shift pos2) {
                return alpha * L1Smoothing(pos1.x, pos1.y, pos2.x, pos2.y);
        };
    } else if(smoothing_func == SmoothingFunction::kBetaL1) {
        options.smoothing = [beta = args.beta_smoothing_param,
                             alpha = args.alpha_smoothing_param] (ImageShift::Shift pos1,
                                                                  ImageShift::Shift pos2) {
            return alpha * std::min(beta, L1Smoothing(pos1.x, pos1.y, pos2.x, pos2.y));
        };
    }
}

