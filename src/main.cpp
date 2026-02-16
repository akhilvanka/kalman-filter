#include "ekf.h"
#include <cstdio>
#include <cmath>
#include <random>
#include <algorithm>

static constexpr double G = 9.80665;
static constexpr double FS = 100.0; // Hz

int main() {
    std::printf("=== 9-State EKF Navigation Demo ===\n\n");

    ekf::EKF filter;
    ekf::NoiseParams noise;
    noise.q_vel   = 0.5;
    noise.r_gps_pos = 9.0;   // 3m GPS noise
    noise.r_gps_vel = 0.04;  // 0.2 m/s GPS velocity noise
    noise.r_baro    = 4.0;   // 2m barometer noise

    // True initial state: hovering then controlled descent
    ekf::StateVec x_true{};
    x_true[2] = -3000.0;  // pos_d (NED down, negative = above ground)
    x_true[5] = 220.0;    // vel_d (positive = downward in NED)
    x_true[6] = 0.02;     // small roll
    x_true[7] = 0.01;     // small pitch

    // Initialize EKF with a noisy estimate
    ekf::StateVec x0 = x_true;
    x0[2] += 20.0;   // 20m position error
    x0[5] += 5.0;    // 5 m/s velocity error
    filter.initialize(x0, 25.0, 10.0, 0.1);

    std::mt19937 rng(42);
    std::normal_distribution<double> imu_noise(0.0, 0.02);
    std::normal_distribution<double> gps_noise(0.0, 3.0);
    std::normal_distribution<double> baro_noise(0.0, 2.0);

    // Simulated thrust for controlled descent (~1g decel)
    double thrust_accel = 2.0; // net upward decel in m/s^2

    std::printf("%-6s %-10s %-10s %-10s %-10s %-10s %-10s\n",
        "t(s)", "TrueAlt", "EKFAlt", "AltErr", "TrueVz", "EKFVz", "PosStd");
    std::printf("%s\n", std::string(70, '-').c_str());

    double t = 0.0;
    double dt = 1.0 / FS;
    int steps = 0;

    while (t < 30.0 && -x_true[2] > 10.0) {
        // True dynamics: decelerate vertically
        x_true[5] = std::max(x_true[5] - thrust_accel * dt, 0.0);
        x_true[2] += x_true[5] * dt;

        // Noisy IMU readings
        ekf::ControlVec imu = {
            imu_noise(rng),
            imu_noise(rng),
            -(thrust_accel - G) + imu_noise(rng), // body accel z (includes gravity)
            imu_noise(rng) * 0.1,
            imu_noise(rng) * 0.1,
            imu_noise(rng) * 0.1
        };

        filter.predict(imu, dt);

        // GPS update at 5 Hz
        if (steps % 20 == 0) {
            ekf::GpsMeasurement gps;
            gps.pos_n = x_true[0] + gps_noise(rng);
            gps.pos_e = x_true[1] + gps_noise(rng);
            gps.pos_d = x_true[2] + gps_noise(rng);
            gps.vel_n = x_true[3];
            gps.vel_e = x_true[4];
            gps.vel_d = x_true[5] + gps_noise(rng) * 0.2;
            filter.update_gps(gps);
        }

        // Baro update at 10 Hz
        if (steps % 10 == 0) {
            ekf::BaroMeasurement baro;
            baro.altitude_m = -x_true[2] + baro_noise(rng);
            filter.update_baro(baro);
        }

        // Print every 2 seconds
        if (steps % 200 == 0) {
            double true_alt = -x_true[2];
            double ekf_alt  = filter.altitude();
            double alt_err  = std::fabs(true_alt - ekf_alt);
            double true_vz  = x_true[5];
            double ekf_vz   = filter.vel_d();

            std::printf("%-6.1f %-10.1f %-10.1f %-10.2f %-10.1f %-10.1f %-10.2f\n",
                t, true_alt, ekf_alt, alt_err, true_vz, ekf_vz, filter.pos_std());
        }

        t += dt;
        ++steps;
    }

    // Final assessment
    double final_alt_err = std::fabs(filter.altitude() - (-x_true[2]));
    double final_vel_err = std::fabs(filter.vel_d() - x_true[5]);
    double final_pos_std = filter.pos_std();

    std::printf("\n=== Final State Estimate ===\n");
    std::printf("  True altitude:      %.2f m\n", -x_true[2]);
    std::printf("  EKF altitude:       %.2f m\n", filter.altitude());
    std::printf("  Position error:     %.2f m (1σ = %.2f m)\n",
        final_alt_err, final_pos_std);
    std::printf("  Velocity error:     %.3f m/s\n", final_vel_err);
    std::printf("  Position std (P):   %.3f m\n", final_pos_std);

    bool pass = final_alt_err < 5.0 * final_pos_std; // error within 5σ
    std::printf("\n[%s] EKF converged: error %.2fm within %.1f sigma\n",
        pass ? "PASS" : "WARN", final_alt_err, final_alt_err / final_pos_std);

    return pass ? 0 : 1;
}
