#pragma once
/// Extended Kalman Filter for 9-state inertial navigation.
///
/// State vector x ∈ ℝ⁹:
///   [0..2]  position (m)         — NED frame
///   [3..5]  velocity (m/s)       — NED frame
///   [6..8]  attitude (roll, pitch, yaw) in rad — small-angle Euler
///
/// Inputs (control):
///   u = [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z] from IMU
///
/// Measurements:
///   GPS position + velocity (H_gps ∈ ℝ⁶ˣ⁹)
///   Barometer altitude     (H_baro ∈ ℝ¹ˣ⁹)
///
/// The EKF is the workhorse of aerospace navigation. Unlike a complementary
/// filter, it maintains a formal uncertainty estimate (covariance P) and
/// weights innovations by how surprising they are — GPS updates have more
/// pull when the IMU has been integrating for a long time.
///
/// Implementation uses Eigen-free fixed-size matrices templated on state
/// dimension N and measurement dimension M to stay header-only.
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <optional>

namespace ekf {

static constexpr int N = 9; // state dimension
static constexpr int U = 6; // control/input dimension

// ---- Fixed-size matrix types ----

template<int Rows, int Cols>
struct Mat {
    std::array<double, Rows * Cols> data{};
    double& at(int r, int c)       { return data[r * Cols + c]; }
    double  at(int r, int c) const { return data[r * Cols + c]; }
    void zero() { data.fill(0.0); }
    static Mat identity() {
        static_assert(Rows == Cols);
        Mat m; m.zero();
        for (int i = 0; i < Rows; ++i) m.at(i,i) = 1.0;
        return m;
    }
};

using StateVec  = std::array<double, N>;
using ControlVec = std::array<double, U>;
using MatNN     = Mat<N, N>;

// ---- Matrix operations ----

template<int R, int K, int C>
Mat<R,C> matmul(const Mat<R,K>& A, const Mat<K,C>& B) {
    Mat<R,C> out; out.zero();
    for (int r = 0; r < R; ++r)
        for (int k = 0; k < K; ++k)
            for (int c = 0; c < C; ++c)
                out.at(r,c) += A.at(r,k) * B.at(k,c);
    return out;
}

template<int R, int C>
Mat<C,R> transpose(const Mat<R,C>& A) {
    Mat<C,R> out;
    for (int r = 0; r < R; ++r)
        for (int c = 0; c < C; ++c)
            out.at(c,r) = A.at(r,c);
    return out;
}

template<int N_>
Mat<N_,N_> mat_add(const Mat<N_,N_>& A, const Mat<N_,N_>& B) {
    Mat<N_,N_> out;
    for (int i = 0; i < N_*N_; ++i) out.data[i] = A.data[i] + B.data[i];
    return out;
}

template<int N_>
Mat<N_,N_> mat_sub(const Mat<N_,N_>& A, const Mat<N_,N_>& B) {
    Mat<N_,N_> out;
    for (int i = 0; i < N_*N_; ++i) out.data[i] = A.data[i] - B.data[i];
    return out;
}

// Numerical Gauss-Jordan matrix inverse (small matrices only)
template<int N_>
bool mat_inv(const Mat<N_,N_>& A, Mat<N_,N_>& Ainv) {
    Mat<N_, 2*N_> aug;
    // Build augmented matrix [A | I]
    for (int r = 0; r < N_; ++r) {
        for (int c = 0; c < N_; ++c) aug.at(r, c) = A.at(r, c);
        aug.at(r, N_ + r) = 1.0;
    }
    for (int col = 0; col < N_; ++col) {
        // Partial pivot
        int pivot = col;
        for (int r = col+1; r < N_; ++r)
            if (std::fabs(aug.at(r,col)) > std::fabs(aug.at(pivot,col))) pivot = r;
        if (std::fabs(aug.at(pivot,col)) < 1e-12) return false;
        for (int c = 0; c < 2*N_; ++c) std::swap(aug.at(col,c), aug.at(pivot,c));

        double s = 1.0 / aug.at(col,col);
        for (int c = 0; c < 2*N_; ++c) aug.at(col,c) *= s;

        for (int r = 0; r < N_; ++r) {
            if (r == col) continue;
            double f = aug.at(r,col);