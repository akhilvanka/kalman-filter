#include "ekf.h"
#include <cstdio>
#include <cmath>
#include <cassert>

#define PASS(n) std::printf("PASS: %s\n", n)
#define FAIL(n,m) do { std::fprintf(stderr, "FAIL: %s — %s\n", n, m); std::exit(1); } while(0)
#define ASSERT_NEAR(a,b,t) do { \
    if (std::fabs((double)(a)-(double)(b))>(double)(t)) { \
        std::fprintf(stderr,"FAIL: |%.6f-%.6f|=%.6f>%.6f @%d\n", \
            (double)(a),(double)(b),std::fabs((double)(a)-(double)(b)),(double)(t),__LINE__); \
        std::exit(1); } } while(0)

// ---- Matrix operations ----

void test_matrix_identity() {
    auto I = ekf::MatNN::identity();
    for (int r = 0; r < ekf::N; ++r)
        for (int c = 0; c < ekf::N; ++c) {
            double expected = (r == c) ? 1.0 : 0.0;
            ASSERT_NEAR(I.at(r,c), expected, 1e-12);
        }
    PASS("matrix_identity");
}

void test_matrix_inverse() {
    ekf::Mat<3,3> A;
    A.at(0,0)=4; A.at(0,1)=7; A.at(0,2)=2;
    A.at(1,0)=3; A.at(1,1)=6; A.at(1,2)=1;
    A.at(2,0)=2; A.at(2,1)=5; A.at(2,2)=3;

    ekf::Mat<3,3> Ainv;
    bool ok = ekf::mat_inv<3>(A, Ainv);
    if (!ok) FAIL("matrix_inverse", "inversion failed");

    // Verify A * A^{-1} = I
    auto I = ekf::matmul<3,3,3>(A, Ainv);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            ASSERT_NEAR(I.at(r,c), (r==c)?1.0:0.0, 1e-10);
    PASS("matrix_inverse");
}

void test_matrix_transpose() {
    ekf::Mat<2,3> A;
    A.at(0,0)=1; A.at(0,1)=2; A.at(0,2)=3;
    A.at(1,0)=4; A.at(1,1)=5; A.at(1,2)=6;
    auto At = ekf::transpose(A);
    ASSERT_NEAR(At.at(0,0), 1.0, 1e-12);
    ASSERT_NEAR(At.at(1,0), 2.0, 1e-12);
    ASSERT_NEAR(At.at(2,1), 6.0, 1e-12);
    PASS("matrix_transpose");
}

// ---- EKF tests ----

void test_ekf_initialize() {
    ekf::EKF f;
    if (f.is_initialized()) FAIL("ekf_initialize", "should not start initialized");

    ekf::StateVec x0{};
    x0[2] = -1000.0;
    x0[5] = 100.0;
    f.initialize(x0, 10.0, 5.0, 0.1);

    if (!f.is_initialized()) FAIL("ekf_initialize", "should be initialized after init");
    ASSERT_NEAR(f.altitude(), 1000.0, 1e-9);
    ASSERT_NEAR(f.vel_d(),    100.0,  1e-9);
    PASS("ekf_initialize");
}

void test_ekf_predict_free_fall() {
    // Body convention: az = specific force in body z.
    // Free fall → IMU reads zero (weightless). With az=0, accel_d = -G.
    // vel_d decreases by G*dt (NED down is positive, gravity acts in -vel_d direction
    // when no thrust is commanded and the specific force cancels gravity).
    ekf::EKF f;
    ekf::StateVec x0{}; x0[2] = -1000.0;
    f.initialize(x0, 5.0, 2.0, 0.1);

    ekf::ControlVec imu_free_fall = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double dt = 0.1;
    f.predict(imu_free_fall, dt);

    // Free fall: vel_d changes by -G*dt (accel_d = -G when az=0)
    ASSERT_NEAR(f.vel_d(), -9.80665 * dt, 0.05);
    PASS("ekf_predict_free_fall");
}

void test_ekf_gps_update_reduces_covariance() {
    ekf::EKF f;
    ekf::StateVec x0{};
    x0[2] = -500.0;
    f.initialize(x0, 50.0, 10.0, 0.1); // large initial uncertainty

    double p_before = f.pos_std();

    // GPS update with accurate measurement
    ekf::GpsMeasurement gps{};
    gps.pos_d = -500.0;
    f.update_gps(gps);

    double p_after = f.pos_std();
    if (p_after >= p_before)
        FAIL("ekf_gps_reduces_cov", "covariance should decrease after GPS update");
    PASS("ekf_gps_update_reduces_covariance");
}

void test_ekf_innovation_corrects_state() {
    // Initialize with a 20m altitude error, then feed GPS with true altitude
    ekf::EKF f;
    ekf::StateVec x0{};
    x0[2] = -1020.0; // EKF thinks we're 20m lower (NED: more negative = higher altitude)
    f.initialize(x0, 30.0, 5.0, 0.1);

    double alt_before = f.altitude(); // 1020m (wrong)

    ekf::GpsMeasurement gps{};
    gps.pos_d = -1000.0; // true position
    gps.vel_d = 0.0;
    for (int i = 0; i < 10; ++i) f.update_gps(gps); // multiple updates

    double alt_after = f.altitude();
    // After several GPS updates, should be much closer to true 1000m
    if (std::fabs(alt_after - 1000.0) >= std::fabs(alt_before - 1000.0))
        FAIL("ekf_innovation_corrects_state", "GPS update should move estimate toward truth");
    PASS("ekf_innovation_corrects_state");
}

void test_ekf_baro_update() {
    ekf::EKF f;
    ekf::StateVec x0{};
    x0[2] = -500.0;
    f.initialize(x0, 20.0, 5.0, 0.1);

    // Baro observes pos_d (index 2) — check alt_std(), not pos_std()
    double p_before = f.alt_std();
    ekf::BaroMeasurement baro{ 510.0 }; // baro says 510m, filter says 500m
    f.update_baro(baro);

    double p_after = f.alt_std();
    if (p_after >= p_before)
        FAIL("ekf_baro_update", "altitude covariance should decrease after baro");
    // State should have moved slightly toward 510
    if (f.altitude() <= 500.0)
        FAIL("ekf_baro_update", "estimate should move toward baro reading");
    PASS("ekf_baro_update");
}

void test_ekf_convergence() {
    // Feed noisy but unbiased measurements for 5 seconds.
    // Final position uncertainty should be << initial.
    ekf::EKF f;
    ekf::StateVec x0{};
    x0[2] = -2000.0;
    f.initialize(x0, 100.0, 20.0, 0.5); // large initial uncertainty

    double initial_std = f.pos_std();

    double dt = 0.01;
    ekf::ControlVec imu = {0,0,0,0,0,0};

    for (int step = 0; step < 500; ++step) {
        f.predict(imu, dt);
        if (step % 20 == 0) { // GPS at 5Hz
            ekf::GpsMeasurement gps{};
            gps.pos_d = -2000.0;
            f.update_gps(gps);
        }
    }

    double final_std = f.pos_std();
    if (final_std >= initial_std * 0.5)
        FAIL("ekf_convergence", "uncertainty did not converge after GPS updates");
    PASS("ekf_convergence");
}

int main() {
    std::printf("Running EKF unit tests...\n\n");

    test_matrix_identity();
    test_matrix_inverse();
    test_matrix_transpose();
    test_ekf_initialize();
    test_ekf_predict_free_fall();
    test_ekf_gps_update_reduces_covariance();
    test_ekf_innovation_corrects_state();
    test_ekf_baro_update();
    test_ekf_convergence();

    std::printf("\nAll tests passed.\n");
    return 0;
}
