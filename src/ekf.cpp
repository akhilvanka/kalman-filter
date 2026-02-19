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
    for (int i = 0; i < 3; ++i) P_.at(i,i) = pos_std * pos_std;
    for (int i = 3; i < 6; ++i) P_.at(i,i) = vel_std * vel_std;
    for (int i = 6; i < 9; ++i) P_.at(i,i) = att_std * att_std;
    initialized_ = true;
}

MatNN EKF::build_F(const ControlVec& imu, double dt) const {
    double roll  = x_[6];
    double pitch = std::clamp(x_[7], -1.5, 1.5);
    double cp = std::cos(pitch), sp = std::sin(pitch);
    double cr = std::cos(roll),  sr = std::sin(roll);

    (void)imu;
    MatNN F = MatNN::identity();
    for (int i = 0; i < 3; ++i) F.at(i, i+3) = dt;

    // vel ← R(attitude) * accel, partials w.r.t. roll and pitch
    F.at(3, 6) = ( sr*sp*imu[0] + cr*imu[1] - sr*cp*imu[2]) * dt;
    F.at(3, 7) = (-cr*cp*imu[0]              + cr*sp*imu[2]) * dt;
    F.at(4, 6) = (-cr*sp*imu[0] + sr*imu[1] + cr*cp*imu[2]) * dt;
    F.at(4, 7) = (-sr*cp*imu[0]              + sr*sp*imu[2]) * dt;
    F.at(5, 7) = (-sp*imu[0]    - cp*imu[2])                 * dt;

    for (int i = 6; i < 9; ++i) F.at(i, i) = 1.0;
    return F;
}

MatNN EKF::build_Q(double dt) const {
    MatNN Q; Q.zero();
    double pa = noise_.accel_noise_density  * noise_.accel_noise_density  * dt;
    double pg = noise_.gyro_noise_density   * noise_.gyro_noise_density   * dt;
    double ba = noise_.accel_bias_instability * noise_.accel_bias_instability * dt;
    double bg = noise_.gyro_bias_instability  * noise_.gyro_bias_instability  * dt;
    for (int i = 3; i < 6; ++i) Q.at(i,i) = pa + ba;
    for (int i = 6; i < 9; ++i) Q.at(i,i) = pg + bg;
    return Q;
}

static MatNN mat_mul(const MatNN& A, const MatNN& B) {
    MatNN C; C.zero();
    for (int i = 0; i < N; ++i)
        for (int k = 0; k < N; ++k)
            for (int j = 0; j < N; ++j)
                C.at(i,j) += A.at(i,k) * B.at(k,j);
    return C;
}

static MatNN mat_transpose(const MatNN& A) {
    MatNN T; T.zero();
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            T.at(j,i) = A.at(i,j);
    return T;
}

void EKF::predict(const ControlVec& imu, double dt) {
    if (!initialized_) throw std::runtime_error("EKF not initialized");

    double roll  = x_[6];
    double pitch = std::clamp(x_[7], -1.5, 1.5);
    double yaw   = x_[8];
    double cp = std::cos(pitch), sp = std::sin(pitch);
    double cr = std::cos(roll),  sr = std::sin(roll);
    double cy = std::cos(yaw),   sy = std::sin(yaw);

    // Rotate body-frame accelerations to NED
    double ax_n = (cp*cy)*imu[0] + (sr*sp*cy - cr*sy)*imu[1] + (cr*sp*cy + sr*sy)*imu[2];
    double ay_n = (cp*sy)*imu[0] + (sr*sp*sy + cr*cy)*imu[1] + (cr*sp*sy - sr*cy)*imu[2];
    double az_n = (-sp) *imu[0]  + (sr*cp)           *imu[1] + (cr*cp)           *imu[2];

    x_[0] += x_[3] * dt;
    x_[1] += x_[4] * dt;
    x_[2] += x_[5] * dt;
    x_[3] += ax_n * dt;
    x_[4] += ay_n * dt;
    x_[5] += (az_n - G) * dt;
    x_[6] += imu[3] * dt;
    x_[7] += imu[4] * dt;
    x_[8] += imu[5] * dt;

    MatNN F = build_F(imu, dt);
    MatNN Q = build_Q(dt);
    MatNN Ft = mat_transpose(F);
    P_ = mat_mul(mat_mul(F, P_), Ft);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            P_.at(i,j) += Q.at(i,j);
}

template<int M>
static bool invert(const Mat<M,M>& S, Mat<M,M>& Sinv) {
    Mat<M,M> a = S;
    Sinv = Mat<M,M>::identity();
    for (int col = 0; col < M; ++col) {
        int pivot = -1;
        double best = 0;
        for (int row = col; row < M; ++row)
            if (std::abs(a.at(row,col)) > best) { best = std::abs(a.at(row,col)); pivot = row; }
        if (pivot < 0 || best < 1e-14) return false;
        for (int j = 0; j < M; ++j) {
            std::swap(a.at(col,j), a.at(pivot,j));
            std::swap(Sinv.at(col,j), Sinv.at(pivot,j));
        }
        double inv = 1.0 / a.at(col,col);
        for (int j = 0; j < M; ++j) { a.at(col,j) *= inv; Sinv.at(col,j) *= inv; }
        for (int row = 0; row < M; ++row) {
            if (row == col) continue;
            double f = a.at(row,col);
            for (int j = 0; j < M; ++j) { a.at(row,j) -= f*a.at(col,j); Sinv.at(row,j) -= f*Sinv.at(col,j); }
        }
    }
    return true;
}

void EKF::update_gps(const GpsMeasurement& gps) {
    if (!initialized_) return;

    Mat<6,9> H; H.zero();
    for (int i = 0; i < 3; ++i) H.at(i,   i  ) = 1.0;
    for (int i = 0; i < 3; ++i) H.at(i+3, i+3) = 1.0;

    Mat<6,6> R; R.zero();
    for (int i = 0; i < 3; ++i) R.at(i,   i  ) = gps.pos_std * gps.pos_std;
    for (int i = 0; i < 3; ++i) R.at(i+3, i+3) = gps.vel_std * gps.vel_std;

    Mat<9,6> Ht; Ht.zero();
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 9; ++j)
            Ht.at(j,i) = H.at(i,j);

    // S = H P Ht + R
    Mat<6,6> S; S.zero();
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j) {
            for (int k = 0; k < 9; ++k)
                for (int m = 0; m < 9; ++m)
                    S.at(i,j) += H.at(i,k) * P_.at(k,m) * Ht.at(m,j);
            S.at(i,j) += R.at(i,j);
        }

    Mat<6,6> Sinv;
    if (!invert<6>(S, Sinv)) return;

    // K = P Ht Sinv  [9x6]
    Mat<9,6> K; K.zero();
    for (int i = 0; i < 9; ++i)
        for (int j = 0; j < 6; ++j)
            for (int k = 0; k < 9; ++k)
                for (int m = 0; m < 6; ++m)
                    K.at(i,j) += P_.at(i,k) * Ht.at(k,m) * Sinv.at(m,j);

    std::array<double,6> z = {gps.pos_n, gps.pos_e, gps.pos_d,
                               gps.vel_n, gps.vel_e, gps.vel_d};
    std::array<double,6> innov;
    for (int i = 0; i < 6; ++i) {
        double hx = (i < 3) ? x_[i] : x_[i];
        innov[i] = z[i] - hx;
    }

    for (int i = 0; i < 9; ++i)
        for (int j = 0; j < 6; ++j)
            x_[i] += K.at(i,j) * innov[j];

    MatNN KH; KH.zero();
    for (int i = 0; i < 9; ++i)
        for (int j = 0; j < 9; ++j)
            for (int k = 0; k < 6; ++k)
                KH.at(i,j) += K.at(i,k) * H.at(k,j);

    MatNN I = MatNN::identity();
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            KH.at(i,j) = I.at(i,j) - KH.at(i,j);

    P_ = mat_mul(KH, P_);
}

void EKF::update_baro(double altitude_m, double alt_std) {
    if (!initialized_) return;

    // H = [0 0 1 0 0 0 0 0 0]
    double h = x_[2];
    double innov = altitude_m - h;
    double S = P_.at(2,2) + alt_std * alt_std;
    if (std::abs(S) < 1e-14) return;

    // K = P * Ht / S  (scalar measurement)
    for (int i = 0; i < 9; ++i) {
        double k = P_.at(i,2) / S;
        x_[i] += k * innov;
        for (int j = 0; j < 9; ++j)
            P_.at(i,j) -= k * P_.at(2,j);
    }
}

const StateVec& EKF::state()      const { return x_; }
const MatNN&   EKF::covariance()  const { return P_; }
bool            EKF::initialized() const { return initialized_; }

} // namespace ekf
