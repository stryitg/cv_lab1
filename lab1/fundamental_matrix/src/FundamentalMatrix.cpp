#include <iostream>
#include <algorithm>

#include "FundamentalMatrix.hpp"

FundamentalMatrix::FundamentalMatrix(const FundamentalMatrix::MatchesInternal& internal)
    : _matches(ToMatches(internal))
    , _gen(_rd())
    , _dis(0, _matches.size() - 1) {}
    
std::vector<FundamentalMatrix::Match> FundamentalMatrix::ToMatches(
            const FundamentalMatrix::MatchesInternal& internal) const {
    std::vector<Match> matches;
    const auto& left_image = internal.left_image;
    const auto& right_image = internal.right_image;
    const auto& shifts = internal.shifts;
    
    for(size_t i = 0; i < left_image.rows; ++i) { 
        for(size_t j = 0; j < left_image.cols; ++j) {
            if(IsDense(shifts, i, j)) {
                const auto& shift = shifts.at<cv::Vec3b>(i, j);
                const auto i_d = static_cast<double>(i);
                const auto j_d = static_cast<double>(j);
                matches.push_back({{j_d, i_d, 1.0}, 
                                  {j_d - (shift[1] + internal.min_shift_x), 
                                   i_d - (shift[0] + internal.min_shift_y), 1.0}});
            }
        }
    }
    std::cout << matches.size() << std::endl;
    return matches;
}

bool FundamentalMatrix::IsDense(const cv::Mat& shifts, size_t i, size_t j) {
    const auto& shift = shifts.at<cv::Vec3b>(i, j);
    bool is_dense = (j > 1) && (j < shifts.cols - 2)
                  && (i > 1) && (i < shifts.rows - 2)
                  && (shift[0] != 0 || shift[1] != 0);
    for(size_t k = -2; k <= 2; ++k) {
        for(size_t m = -2; m <= 2; ++m) {
            const auto& shift_ = shifts.at<cv::Vec3b>(i + k, j + m);
            is_dense = is_dense && (shift[0] == shift_[0]) && (shift[1] == shift_[1]);
        }
    }
    return is_dense;
}

Eigen::Matrix3d FundamentalMatrix::Get() const {
    return RANSAC(); 
}

Eigen::Matrix3d FundamentalMatrix::RANSAC() const {
    auto pts = GetSevenRandomPoints();
    auto best = SevenPtsAlgotihm(pts);
    auto best_inliers_count = GetInliersCount(best);
    for(size_t i = 0; i < kIter; ++i) {
        pts = GetSevenRandomPoints();
        const auto matrix = SevenPtsAlgotihm(pts);
        const auto inliers_count = GetInliersCount(matrix);
        if(inliers_count > best_inliers_count) {
            best_inliers_count = inliers_count;
            best = std::move(matrix);
        }
    }
    return best;
}

std::array<FundamentalMatrix::Match, 7> FundamentalMatrix::GetSevenRandomPoints() const {
    const auto random_indexes = GetRandomIndexes();
    std::array<FundamentalMatrix::Match, 7> pts;
    std::transform(random_indexes.begin(), random_indexes.end(),
                   pts.begin(), [this] (const auto& index) {
                       return _matches[index]; 
                   });
    return pts;
}

std::unordered_set<size_t> FundamentalMatrix::GetRandomIndexes() const {
    std::unordered_set<size_t> pts;
    while(pts.size() != 7) {
        pts.insert(_dis(_gen));
    }
    return pts;
}

Eigen::Matrix3d FundamentalMatrix::SevenPtsAlgotihm(
            const std::array<FundamentalMatrix::Match, 7>& pts) const {
    const auto matrix = GetMatrixRepresentation(pts);
    const auto kernel = GetKernel(matrix);
    const auto eq_matrices = ToEqMatrices(kernel);
    for(const auto& mat: eq_matrices) {
        if(GetRank(mat) == 2) {
            return mat;
        }
    }
    const auto polynomial = GetFundamentalMatrixPolynomial(eq_matrices);
    const auto roots = Solve(polynomial);
    for(const auto& root: roots) {
        if(root.imag() == 0.0) {
            return eq_matrices[0] + root.real() * eq_matrices[1];
        }
    }
    
    throw std::runtime_error("No real roots");
}

FundamentalMatrix::Matrix7x9d FundamentalMatrix::GetMatrixRepresentation(
            const std::array<Match, 7>& pts) const {
    Matrix7x9d matrix;
    for(size_t i = 0; i < 7; ++i) {
        const auto& left = pts[i].left;
        const auto& right = pts[i].right;
        for(size_t j = 0; j < left.size(); ++j) {
            for(size_t k = 0; k < right.size(); ++k) {
                matrix(i, 3 * j + k) = left(j) * right(k);
            }
        }
    }
    return matrix;
}

Eigen::MatrixXd FundamentalMatrix::GetKernel(
            const FundamentalMatrix::Matrix7x9d& matrix) const {
    const Eigen::FullPivLU<Matrix7x9d> lu(matrix);
    return lu.kernel();
}

std::array<Eigen::Matrix3d, 2> FundamentalMatrix::ToEqMatrices(
            const Eigen::MatrixXd& kernel) const {
    const auto& c1 = kernel.col(0);
    const auto& c2 = kernel.col(1);
    const auto a = Eigen::Map<const Eigen::Matrix3d>(c1.data());
    const auto b = Eigen::Map<const Eigen::Matrix3d>(c2.data());
    return {a, b};
}

Eigen::Index FundamentalMatrix::GetRank(
            const Eigen::Matrix3d& matrix) const {
    const Eigen::FullPivLU<Eigen::Matrix3d> lu(matrix);
    return lu.rank();
}

std::array<double, 4> FundamentalMatrix::GetFundamentalMatrixPolynomial(
            const std::array<Eigen::Matrix3d, 2>& eq_matrices) const {
    const auto& a = eq_matrices[0];
    const auto& b = eq_matrices[1];
    
    const auto det_a = a.determinant();
    const auto det_b = b.determinant();
    const auto inv_a_times_b = a.inverse() * b;
    const auto inv_b_times_a = b.inverse() * a;
    
    return {det_a, det_a * inv_a_times_b.trace(), 
            det_b * inv_b_times_a.trace(), det_b};
}

// a hack; representing polynomial as matrix and finding its eigenvalues
FundamentalMatrix::Roots FundamentalMatrix::Solve(
            const std::array<double, 4>& polynomial) const {
    const auto a = -polynomial[0] / polynomial[3];
    const auto b = -polynomial[1] / polynomial[3]; 
    const auto c = -polynomial[2] / polynomial[3];
    
    Eigen::Matrix3d matrix;
    matrix << 0, 1, 0,
              0, 0, 1,
              a, b, c;
    Eigen::EigenSolver<Eigen::Matrix3d> solver(matrix);
    return solver.eigenvalues();
}

size_t FundamentalMatrix::GetInliersCount(
            const Eigen::Matrix3d& fundamental_matrix) const {
    size_t count = 0;
    for(const auto& match: _matches) {
        const auto val = match.right.transpose() * fundamental_matrix * match.left;
        if(std::abs(val) < kEps) {
            ++count;
        }
    }
    assert(count >= 7);
    return count;
}

Eigen::Vector3d FundamentalMatrix::GetEpipole(const Eigen::Matrix3d& fm) {
    const Eigen::FullPivLU<Eigen::Matrix3d> lu(fm);
    const auto epipole = lu.kernel();
    if(std::abs(epipole(2)) > 0.00001) {
        return epipole / epipole(2);
    } else {
        std::cout << epipole << std::endl;
        return epipole * 1000000;
    }
}

