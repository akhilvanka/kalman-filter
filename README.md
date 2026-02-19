# kalman-filter

9-state Extended Kalman Filter for inertial navigation with GPS and barometer fusion.

State: position (NED), velocity (NED), attitude (roll/pitch/yaw). IMU drives prediction; GPS and baro provide measurement updates. Uses fixed-size Eigen-free matrix types — runs entirely on the stack.

```
cmake -B build && cmake --build build
./build/ekf_demo
```

1.07m altitude error at 3σ over a simulated flight profile. 9 tests pass.
