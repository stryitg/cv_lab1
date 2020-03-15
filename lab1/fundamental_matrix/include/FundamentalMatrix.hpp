#pragma once

#include <cstdint>
#include <random>
#include <unordered_set>

#include <opencv2/core/core.hpp>
#include <Core>
#include <LU>
#include <Eigenvalues> 

class FundamentalMatrix {
public:
    using Matrix7x9d = Eigen::Matrix<double, 7, 9>;
    using Roots = Eigen::EigenSolver<Eigen::Matrix3d>::EigenvalueType;
    
    struct MatchesInternal {
        cv::Mat left_image;
        cv::Mat right_image;
        cv::Mat shifts;
        int32_t min_shift_x;
        int32_t min_shift_y;
    };
    
    struct Match {
        Eigen::Vector3d left;
        Eigen::Vector3d right;
    };
    
    
    FundamentalMatrix(const MatchesInternal& internal);
    Eigen::Matrix3d Get() const;
    static Eigen::Vector3d GetEpipole(const Eigen::Matrix3d& fm);
    
private:
    std::vector<Match> ToMatches(const MatchesInternal& internal) const;
    static bool IsDense(const cv::Mat& shifts, size_t i, size_t j);
    
    Eigen::Matrix3d RANSAC() const;
    
    std::array<Match, 7> GetSevenRandomPoints() const;
    std::unordered_set<size_t> GetRandomIndexes() const;
    
    Eigen::Matrix3d SevenPtsAlgotihm(const std::array<Match, 7>& pt) const;
    Matrix7x9d GetMatrixRepresentation(const std::array<Match, 7>& pt) const;
    Eigen::MatrixXd GetKernel(const Matrix7x9d& matrix) const;
    std::array<Eigen::Matrix3d, 2> ToEqMatrices(const Eigen::MatrixXd& kernel) const;
    Eigen::Index GetRank(const Eigen::Matrix3d& matrix) const;
    std::array<double, 4> GetFundamentalMatrixPolynomial(
                const std::array<Eigen::Matrix3d, 2>& eq_matrices) const;
    Roots Solve(const std::array<double, 4>& polynomial) const;
    
    size_t GetInliersCount(const Eigen::Matrix3d& fundamental_matrix) const;
    
private:
    static constexpr uint32_t kIter = 100'000;
    static constexpr double kEps = 0.01;
    
    std::vector<Match> _matches;
    
    mutable std::random_device _rd;
    mutable std::mt19937 _gen;
    mutable std::uniform_int_distribution<size_t> _dis;
 
};