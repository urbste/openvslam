#ifndef OPENVSLAM_IMU_UTILS_H
#define OPENVSLAM_IMU_UTILS_H

#include "openvslam/type.h"
#include "openvslam/imu/data.h"

namespace openvslam {
namespace imu {

unsigned int integrate_gyro_for_rotation(
        const std::vector<openvslam::imu::data>& imu_data,
        const double start_timestamp,
        const double end_timestamp,
        const double time_offset_gyro_to_cam,
        Mat33_t& integrated_rotation,
        const Mat33_t& rotation_gyro_to_cam = Mat33_t::Identity(),
        const Vec3_t& imu_bias = Vec3_t::Zero(),
        const unsigned int last_gyro_start_idx = 0);

}
}

#endif
