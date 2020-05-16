#pragma once

#include <Eigen/Core>

struct Rectification {
    Eigen::Matrix3d left;
    Eigen::Matrix3d right;
};

struct ImgShape {
    size_t height;
    size_t width;
};

struct Match {
    Eigen::Vector3d left;
    Eigen::Vector3d right;
};

struct TransformedMatches {
    std::vector<Eigen::Vector3d> left;
    std::vector<double> right;
};

struct Eq {
    Eigen::Matrix3d m;
    Eigen::Vector3d v;
};

Rectification GetRectification(const Eigen::Matrix3d& fm, const ImgShape& shp,
                               const std::vector<Match>& matches);
Eigen::Matrix3d GetRightRectification(const Eigen::Matrix3d& fm,
                                      const ImgShape& shp);
Eigen::Matrix3d GetShiftMatrix(const ImgShape& shp);
Eigen::Matrix3d GetRotationMatrix(const Eigen::Matrix3d& fm,
                                  const Eigen::Matrix3d& shift);
Eigen::Matrix3d GetProjectionMatrix(const Eigen::Matrix3d& fm,
                                    const Eigen::Matrix3d& affine);
Eigen::Matrix3d GetInvShift(const ImgShape& shp);
Eigen::Matrix3d GetGeneralShiftMatrix(const ImgShape& shp, int sign);
Eigen::Matrix3d GetLeftRectification(const Eigen::Matrix3d& fm,
                                     const Eigen::Matrix3d& right,
                                     const std::vector<Match>& matches);
Eigen::Vector3d GetRightEpipole(const Eigen::Matrix3d& fm);
Eigen::Matrix3d GetBestRectification(const Eigen::Matrix3d& M,
                                     const Eigen::Matrix3d& right,
                                     const std::vector<Match>& matches);
Eigen::Matrix3d GetCrossProductMatrix(const Eigen::Vector3d& v);
Eigen::Vector3d GetBestParams(const Eigen::Matrix3d& M,
                              const Eigen::Matrix3d& right,
                              const std::vector<Match>& matches);
TransformedMatches GetTransformedMatches(const Eigen::Matrix3d& M,
                                         const Eigen::Matrix3d& right,
                                         const std::vector<Match>& matches);
std::vector<Eigen::Vector3d> GetLeftTransformedMatches(
                                     const Eigen::Matrix3d& M,
                                     const std::vector<Match>& matches);
std::vector<double> GetRightTransformedMatches(
                                     const Eigen::Matrix3d& right,
                                     const std::vector<Match>& matches);;
Eigen::Vector3d ProjectiveMult(const Eigen::Matrix3d& m, 
                               const Eigen::Vector3d& v);
Eq GetParamsEquation(const TransformedMatches& matches);  
                             