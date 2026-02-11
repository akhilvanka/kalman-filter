#include "ekf.h"
#include <cmath>
#include <stdexcept>

namespace ekf {

static constexpr double G = 9.80665;

EKF::EKF(const NoiseParams& noise) : noise_(noise) {
    x_.fill(0.0);
    P_.zero();
}

void EKF::initialize(const StateVec& x0, double pos_std, double vel_std, double att_std) {
    x_ = x0;
    P_.zero();
    for (int i = 0; i < 3; ++i) P_.at(i,i)   = pos_std * pos_std;
    for (int i = 3; i < 6; ++i) P_.at(i,i)   = vel_std * vel_std;
    for (int i = 6; i < 9; ++i) P_.at(i,i)   = att_std * att_std;
    initialized_ = true;
}

MatNN EKF::build_F(const ControlVec& imu, double dt) const {
    // State transition Jacobian for x(k+1) = f(x(k), u)
    // Linearized around current state x_
    //
    // pos += vel * dt
    // vel += R(attitude) * accel_body * dt
    // att += omega_body * dt  (small-angle, simplified)
    //
    // F = df/dx = I + (∂f/∂x) * dt (first-order)

    double roll  = x_[6], pitch = x_[7]; // yaw doesn't affect gravity projection here
    double ax    = imu[0], ay = imu[1], az = imu[2];

    // Partials of velocity w.r.t. attitude (from R(att)*accel_body rotation)
    // dvel_n/droll  = -(cos(roll)*sin(pitch)*ax + sin(roll)*ay + cos(roll)*cos(pitch)*az) NOT simplified
    // For a linearized EKF we use first-order approximation
    double cp = std::cos(pitch), sp = std::sin(pitch);
    double cr = std::cos(roll),  sr = std::sin(roll);

    MatNN F = MatNN::identity();

    // pos += vel * dt
    for (int i = 0; i < 3; ++i) F.at(i, i+3) = dt;

    // vel_n += (d/droll)(R*a)*dt and (d/dpitch)(R*a)*dt