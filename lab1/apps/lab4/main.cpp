#include <vector>
#include <algorithm>
#include <iostream>

#include <boost/program_options.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <Eigen/SVD>

#include "FundamentalMatrix.hpp"

namespace po = boost::program_options;

struct Args {
    std::string left_image_path;
    std::string right_image_path;
    std::string shift_map_path;
};

struct ImgShape {
    size_t height;
    size_t width;
};

struct CameraParams {
    double pixel_size;
    double camera_resolution_x;
    double camera_resolution_y;
    double focal_length;
};

struct EsentialMatrixDecomposition {
    Eigen::Matrix3d R;
    Eigen::Vector3d c;
};

void ParseCommandLine(int argc, char** argv,
                      FundamentalMatrix::MatchesInternal& internal);
Eigen::Matrix3d GetEssentialMatrix(const Eigen::Matrix3d& fm,
                                  const CameraParams& params,
                                  const ImgShape& shape);
Eigen::Matrix3d GetCameraIntrinsicsMatrix(const CameraParams& params,
                                          const ImgShape& shape);
Eigen::Matrix3d CheckEssentialMatrix(const Eigen::Matrix3d& E);
Eigen::Matrix3d GetClosestEsentialMatrix(const Eigen::Matrix3d& m);
EsentialMatrixDecomposition GetEssentialMatrixDecomposition(const Eigen::Matrix3d& E);
Eigen::Matrix3d GetCrossProductMatrix(const Eigen::Vector3d& v);

int main(int argc, char** argv) {
    FundamentalMatrix::MatchesInternal internal;
    ParseCommandLine(argc, argv, internal);
    FundamentalMatrix fm(internal);
    const auto f = fm.Get();
    std::cout << "FundamentalMatrix:" << std::endl;
    std::cout << f << std::endl;    
    const CameraParams camera_params_redmi_4x {
        .pixel_size = 0.001127,
        .camera_resolution_x = 4160,
        .camera_resolution_y = 3120,
        .focal_length = 4.12
    };
    
    const auto E = GetEssentialMatrix(f, camera_params_redmi_4x,
                                     {.height = (size_t) internal.left_image.rows,
                                      .width = (size_t) internal.left_image.cols});
    std::cout << "Essential:" << std::endl;
    std::cout << E << std::endl;
    
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(E);
    std::cout << "Singular values:" << std::endl;
    std::cout << svd.singularValues() << std::endl;;
    
    std::cout << "2EE^tE - trace(EE^t)E :" << std::endl;
    std::cout << CheckEssentialMatrix(E) << std::endl;
    
    const auto Em =  GetClosestEsentialMatrix(E);
    std::cout << "Best fit essential matrix: " << std::endl;
    std::cout << Em << std::endl;
    
    std::cout << "2EE^tE - trace(EE^t)E new:" << std::endl;
    std::cout << CheckEssentialMatrix(Em) << std::endl; 

    const auto R_c = GetEssentialMatrixDecomposition(Em);
    std::cout << "R[c]x decmposition: " << std::endl;
    std::cout << "R: " << std::endl << R_c.R << std::endl;
    std::cout << "c: " << std::endl << R_c.c << std::endl;
    
    std::cout << "R[c]x:" << std::endl;
    std::cout << R_c.R * GetCrossProductMatrix(R_c.c) << std::endl;
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

Eigen::Matrix3d GetEssentialMatrix(const Eigen::Matrix3d& fm,
                                  const CameraParams& params,
                                  const ImgShape& shape) {
    const auto K = GetCameraIntrinsicsMatrix(params, shape);
    return K.transpose() * fm * K;
}

Eigen::Matrix3d GetCameraIntrinsicsMatrix(const CameraParams& params,
                                          const ImgShape& shape) {
    const double pixel_sz_x = params.pixel_size * params.camera_resolution_x
                            / shape.width;
    const double pixel_sz_y = params.pixel_size * params.camera_resolution_y
                            / shape.height;
    const double fx = params.focal_length / pixel_sz_x;
    const double fy = params.focal_length / pixel_sz_y;
    Eigen::Matrix3d K;
    K << fx, 0.0,  shape.width / 2.0,
         0,  fy,   shape.height / 2.0,
         0,  0,  1;
    return K;          
}

Eigen::Matrix3d CheckEssentialMatrix(const Eigen::Matrix3d& E) {
    return 2 * E * E.transpose() * E - (E.transpose() * E).trace() * E;
}

Eigen::Matrix3d GetClosestEsentialMatrix(const Eigen::Matrix3d& m) {
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(m, Eigen::ComputeThinU | Eigen::ComputeThinV);
    auto sv = svd.singularValues();
    sv(2) = 0.0;
    const auto tmp = (sv(0) + sv(1)) / 2;
    sv(0) = tmp;
    sv(1) = tmp;
    Eigen::DiagonalMatrix<double, 3> D(sv); 
    return svd.matrixU() * D * svd.matrixV().transpose();
}

EsentialMatrixDecomposition GetEssentialMatrixDecomposition(const Eigen::Matrix3d& E) {
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(E, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const auto s = svd.singularValues()(0);
    Eigen::Matrix3d m;
    m <<  0, 1, 0,
         -1, 0, 0,
          0, 0, 1;
    const auto W = svd.matrixU() * m;
    const Eigen::Matrix3d V = svd.matrixV();
    const auto det_v = V.determinant();
    const auto det_w = W.determinant();
    return EsentialMatrixDecomposition{.R = W * V.transpose() * det_v * det_w,
                                       .c = s * V.col(2) * det_w};
}

Eigen::Matrix3d GetCrossProductMatrix(const Eigen::Vector3d& v) {
    Eigen::Matrix3d m;
    m << 0, -v(2), v(1),
         v(2), 0, -v(0),
         -v(1), v(0), 0;
    return m;
}