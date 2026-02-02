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
