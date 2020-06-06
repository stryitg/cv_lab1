#include <vector>
#include <algorithm>
#include <iostream>
#include <array>

#include <boost/program_options.hpp>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <Eigen/Core>
#include <Eigen/LU>

namespace po = boost::program_options;

struct Args {
    std::string left_image_path;
    std::string central_image_path;
    std::string right_image_path;
};

struct Images {
    cv::Mat left;
    cv::Mat central;
    cv::Mat right;
};

struct Match {
    Eigen::Vector3d p_from;
    Eigen::Vector3d p_into;
};

struct Homographies {
    Images imgs;
    Eigen::Matrix3d left_to_central;
    Eigen::Matrix3d right_to_central;
};

struct Projection {
    cv::Mat img;
    Eigen::Matrix3d mat;
};

constexpr size_t kCanvasSizeY = 960;
constexpr size_t kCanvasSizeX = 2 * 1280;


using Matrix8x9d = Eigen::Matrix<double, 8, 9>;
using Vector9d = Eigen::Matrix<double, 9, 1>;

Args ParseCommandLine(int argc, char** argv);
Images OpenImages(const Args& args);
cv::Mat OpenImage(const std::string& path);

Eigen::Matrix3d GetHomography(const std::array<Match, 4>& matches);
Matrix8x9d GetHomographyEquation(const std::array<Match, 4>& matches);
Vector9d SolveForHomography(const Matrix8x9d& matrix);
Eigen::Matrix3d Reshape(const Vector9d& v);

void DrawHomographies(const Homographies& data);
cv::Mat ProjectOntoCanvas(const std::vector<Projection>& data);

Eigen::Vector3d ProjectiveMult(const Eigen::Matrix3d& m, 
                               const Eigen::Vector3d& v);
bool IsInImage(const Eigen::Vector3d& v, const cv::Mat& img);

int main(int argc, char** argv) {
    // from -- left; into -- central
    const std::array<Match, 4> m1 = {Match{.p_from = Eigen::Vector3d{638, 256, 1}, .p_into = Eigen::Vector3d{364, 262, 1}},
                                     Match{.p_from = Eigen::Vector3d{1232, 576, 1}, .p_into = Eigen::Vector3d{922, 578, 1}},
                                     Match{.p_from = Eigen::Vector3d{952, 474, 1}, .p_into = Eigen::Vector3d{678, 486, 1}},
                                     Match{.p_from = Eigen::Vector3d{412, 868, 1}, .p_into = Eigen::Vector3d{112, 916, 1}}};   
                                     
    // from -- right; into -- central
    const std::array<Match, 4> m2 = {Match{.p_from = Eigen::Vector3d{38, 270, 1}, .p_into = Eigen::Vector3d{364, 262, 1}},
                                     Match{.p_from = Eigen::Vector3d{398, 510, 1}, .p_into = Eigen::Vector3d{678, 486, 1}},
                                     Match{.p_from = Eigen::Vector3d{902, 800, 1}, .p_into = Eigen::Vector3d{1212, 818, 1}},
                                     Match{.p_from = Eigen::Vector3d{878, 404, 1}, .p_into = Eigen::Vector3d{1194, 392, 1}}};

    try {
        const auto args = ParseCommandLine(argc, argv);
        const auto imgs = OpenImages(args);
        const Eigen::Matrix3d h1 = GetHomography(m1);
        std::cout << "Homography left into central: " << std::endl << h1 << std::endl;
        const Eigen::Matrix3d h2 = GetHomography(m2);
        std::cout << "Homography right into central: " << std::endl << h2 << std::endl;
        DrawHomographies({.imgs = imgs, .left_to_central = h1, .right_to_central = h2});
    } catch (const std::exception& ex) {
        std::cout << ex.what() << std::endl;
    }
}

Args ParseCommandLine(int argc, char** argv) {
    Args args;
    
    po::options_description desc("Options");
    desc.add_options()
        ("left-image", po::value<std::string>(&args.left_image_path)->required(), "path to left image")
        ("central-image", po::value<std::string>(&args.central_image_path)->required(), "path to central image")
        ("right-image", po::value<std::string>(&args.right_image_path)->required(), "path to right image")
        ("help", "produces help message");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    
    if(vm.count("help")) {
        std::cout << desc << std::endl;
        std::cout << "Usage:" << std::endl 
        << "./lab1 --left-image limg.png --c --right-image rimg.png" << std::endl;
        return args;
    }
    po::notify(vm);
    
    return args;
}

Images OpenImages(const Args& args) {
    const auto imgs = Images{
        .left = OpenImage(args.left_image_path),
        .central = OpenImage(args.central_image_path),
        .right = OpenImage(args.right_image_path)
    };
    return imgs;
}

cv::Mat OpenImage(const std::string& path) {
    const auto img = cv::imread(path, cv::IMREAD_COLOR);
    if(!img.data) {
        throw std::runtime_error("couldn't open " + path);
    }
    
    return img;
}

Eigen::Matrix3d GetHomography(const std::array<Match, 4>& matches) {
    const auto matrix = GetHomographyEquation(matches);
    const auto homography = SolveForHomography(matrix);
    return Reshape(homography);
}

Matrix8x9d GetHomographyEquation(const std::array<Match, 4>& matches) {
    Matrix8x9d m;
    for (size_t i = 0; i < matches.size(); ++i) {
        const auto& x = matches[i].p_into;
        const auto& x_ = matches[i].p_from;
        m.row(2 * i) = Vector9d{-x_[0], -x_[1], -x_[2], 0, 0, 0, x[0] * x_[0], x[0] * x_[1], x[0] * x_[2]}; 
        m.row(2 * i + 1) = Vector9d{0, 0, 0, -x_[0], -x_[1], -x_[2], x[1] * x_[0], x[1] * x_[1], x[1] * x_[2]}; 
    }
    return m;
}

Vector9d SolveForHomography(const Matrix8x9d& matrix) {
    const Eigen::FullPivLU<Eigen::MatrixXd> lu(matrix);
    return lu.kernel().col(0);
}

Eigen::Matrix3d Reshape(const Vector9d& v) {
    const auto m = Eigen::Map<const Eigen::Matrix3d>(v.data());
    return m.transpose();
}

void DrawHomographies(const Homographies& data) {
    Eigen::Matrix3d T;
    T << 1, 0, (kCanvasSizeX - data.imgs.left.cols) / 2.0,
         0, 1, 0,
         0, 0, 1;
    const std::vector data_ = {Projection{.img = data.imgs.left, .mat = T * data.left_to_central},
                               Projection{.img = data.imgs.central, .mat = T},
                               Projection{.img = data.imgs.right, .mat = T * data.right_to_central}};
    const auto big_img = ProjectOntoCanvas(data_);
    cv::Mat img;
    cv::resize(big_img, img, cv::Size(), 0.5, 0.75);
    cv::namedWindow("Homography", cv::WINDOW_AUTOSIZE);
    cv::imshow("Homography", img);
    cv::waitKey(0);
}

cv::Mat ProjectOntoCanvas(const std::vector<Projection>& data) {
    cv::Mat canvas(cv::Size{kCanvasSizeX, kCanvasSizeY}, CV_8UC3);
    for (size_t i = 0; i < canvas.rows; ++i) {
        for (size_t j = 0; j < canvas.cols; ++j) {
            size_t inliers = 0;
            Eigen::Vector3d pixel = {0.0, 0.0, 0.0};
            Eigen::Vector3d pos = {(double) j, (double) i, 1.0};
            for (const auto& d : data) {
                const auto img_pos = ProjectiveMult(d.mat.inverse(), pos);
                if (IsInImage(img_pos, d.img)) {
                    const auto val = d.img.at<cv::Vec3b>(img_pos(1), img_pos(0));
                    pixel += Eigen::Vector3d{(double) val[0], (double) val[1], (double) val[2]}; 
                    ++inliers;
                }
            }
            if (inliers != 0) {
                pixel /= inliers;
            }
            canvas.at<cv::Vec3b>(i, j) = cv::Vec3b{(uint8_t) pixel(0), (uint8_t) pixel(1), (uint8_t) pixel(2)};
        }
    }
    return canvas;
}

Eigen::Vector3d ProjectiveMult(const Eigen::Matrix3d& m, 
                               const Eigen::Vector3d& v) {
    Eigen::Vector3d res = m * v;
    if(res(2) == 0) {
        res *= 1'000'000;
    } else {
        res /= res(2); 
    }
    return res;
}

bool IsInImage(const Eigen::Vector3d& v, const cv::Mat& img) {
    return 0 <= v(0) && v(0) <= img.cols &&
           0 <= v(1) && v(1) <= img.rows;
}

// DrawHomographies(cv::Mat& canvas, const cv::Mat& img, const Eigen::Matrix3d& P) {
// 
// }