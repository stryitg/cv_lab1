#include <algorithm>
#include <iostream>
#include <Eigen/LU>
#include <Eigen/Dense>

#include "Rectification.hpp"

Rectification GetRectification(const Eigen::Matrix3d& fm, const ImgShape& shp,
                               const std::vector<Match>& matches) {
    const auto right = GetRightRectification(fm, shp);
    const auto epipole = GetRightEpipole(fm);
    const auto left = GetLeftRectification(fm, right, matches);
    const Eigen::FullPivLU<Eigen::Matrix3d> lu(fm);
    const auto epx = lu.kernel();
    return Rectification{.left = std::move(left), .right = std::move(right)};
}

Eigen::Matrix3d GetRightRectification(const Eigen::Matrix3d& fm, 
                                      const ImgShape& shp) {                       
    const auto shift = GetShiftMatrix(shp);
    const auto rotation = GetRotationMatrix(fm, shift);
    const auto affine = rotation * shift;
    const auto projection = GetProjectionMatrix(fm, affine);
    const auto inv_shift = GetInvShift(shp);
    return inv_shift * projection * affine;
}

Eigen::Matrix3d GetShiftMatrix(const ImgShape& shp) {
    return GetGeneralShiftMatrix(shp, -1);
}

Eigen::Matrix3d GetInvShift(const ImgShape& shp) {
    return GetGeneralShiftMatrix(shp, 1);
}

Eigen::Matrix3d GetGeneralShiftMatrix(const ImgShape& shp, int sign) {
    Eigen::Matrix3d m;
    m << 1, 0, /*sign * shp.height / 2.0*/ 0, // works better this way
         0, 1, /*sign * shp.width / 2.0*/ 0,
         0, 0, 1;
    return m;
}

Eigen::Matrix3d GetRotationMatrix(const Eigen::Matrix3d& fm,
                                  const Eigen::Matrix3d& shift) {
    const auto ep = shift * GetRightEpipole(fm);
    const double tg = ep(1) / ep(0);
    const double sin = -tg / std::sqrt(1 + tg * tg);
    const double cos = 1 / std::sqrt(1 + tg * tg);
    
    Eigen::Matrix3d m;
    m << cos,  sin, 0,
         -sin, cos, 0,
         0,    0,   1;
    return m;
}

Eigen::Matrix3d GetProjectionMatrix(const Eigen::Matrix3d& fm,
                                    const Eigen::Matrix3d& affine) {
    const auto ep = GetRightEpipole(fm);
    const auto x = (affine * ep)(0);
    const auto v = -ep(2) / x;
    Eigen::Matrix3d m;
    m << 1, 0, 0,
         0, 1, 0,
         v, 0, 1;
    return m;
}

Eigen::Matrix3d GetLeftRectification(const Eigen::Matrix3d& fm,
                                     const Eigen::Matrix3d& right,
                                     const std::vector<Match>& matches) {
    const auto ep = GetRightEpipole(fm);
    const auto ep_cross_prod = GetCrossProductMatrix(ep);
    
    Eigen::Matrix3d M = right * ep_cross_prod * fm;
    M.row(0) = M.row(1).cross(M.row(2));
    return GetBestRectification(M, right, matches);
}

Eigen::Vector3d GetRightEpipole(const Eigen::Matrix3d& fm) {
    const Eigen::FullPivLU<Eigen::Matrix3d> lu(fm.transpose());
    const auto ep = lu.kernel();
    return ep / ep(2);
}

Eigen::Matrix3d GetCrossProductMatrix(const Eigen::Vector3d& v) {
    Eigen::Matrix3d m;
    m << 0, -v(2), v(1),
         v(2), 0, -v(0),
         -v(1), v(0), 0;
    return m;
}

Eigen::Matrix3d GetBestRectification(const Eigen::Matrix3d& M,
                                     const Eigen::Matrix3d& right,
                                     const std::vector<Match>& matches) {
    const auto bp = GetBestParams(M, right, matches);
    Eigen::Matrix3d m;
    m << bp(0), bp(1), bp(2),
         0,     1,     0,
         0,     0,     1;
    return m * M;
}

Eigen::Vector3d GetBestParams(const Eigen::Matrix3d& M,
                              const Eigen::Matrix3d& right,
                              const std::vector<Match>& matches) {
    const auto transformed_matches = GetTransformedMatches(M, right, matches);
    const auto eq = GetParamsEquation(transformed_matches);
    return eq.m.colPivHouseholderQr().solve(eq.v);
}

TransformedMatches GetTransformedMatches(const Eigen::Matrix3d& M,
                                         const Eigen::Matrix3d& right,
                                         const std::vector<Match>& matches) {
    const auto l_proj = GetLeftTransformedMatches(M, matches);
    const auto r_proj = GetRightTransformedMatches(right, matches);
    return TransformedMatches{.left = l_proj, .right = r_proj};
}

std::vector<Eigen::Vector3d> GetLeftTransformedMatches(
                                     const Eigen::Matrix3d& M,
                                     const std::vector<Match>& matches) {
    std::vector<Eigen::Vector3d>
    l_proj(matches.size());
    std::transform(matches.begin(), matches.end(), l_proj.begin(),
             [&M] (const auto& match) {
                 return ProjectiveMult(M, match.left);
             });
    return l_proj;
}

std::vector<double> GetRightTransformedMatches(
                                     const Eigen::Matrix3d& right,
                                     const std::vector<Match>& matches) {
    std::vector<double> r_proj(matches.size());
    std::transform(matches.begin(), matches.end(), r_proj.begin(),
             [&right] (const auto& match) {
                 return ProjectiveMult(right, match.right)(0);
             });
    return r_proj;
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

Eq GetParamsEquation(const TransformedMatches& matches) {
    Eq eq{.m = Eigen::Matrix3d::Zero(), .v = Eigen::Vector3d::Zero()};
    const auto& l = matches.left;
    const auto& r = matches.right;
    for(size_t j = 0; j < l.size(); ++j) {
        eq.m += l[j] * l[j].transpose();
        eq.v += r[j] * l[j];
    }
    return eq;
}