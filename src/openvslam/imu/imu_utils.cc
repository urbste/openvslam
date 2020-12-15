

#include "openvslam/imu/data.h"
#include "openvslam/imu/imu_utils.h"

#include "g2o/types/slam3d/se3quat.h"

namespace openvslam {
namespace imu {

unsigned int integrate_gyro_for_rotation(
        const std::vector<openvslam::imu::data>& imu_data,
        const double start_timestamp,
        const double end_timestamp,
        const double time_offset_gyro_to_cam,
        Mat33_t& integrated_rotation,
        const Mat33_t& rotation_gyro_to_cam,
        const Vec3_t& imu_bias,
        const unsigned int last_gyro_start_idx) {

    ::g2o::SE3Quat init_rotation(integrated_rotation, Vec3_t::Zero());

    // See also IMU Preintegration on Manifold Eq. 23
    unsigned int new_gyr_idx = last_gyro_start_idx;
    for (unsigned int gyr_idx = last_gyro_start_idx + 1; gyr_idx < imu_data.size(); ++gyr_idx) {
        double gyr_t = imu_data[gyr_idx].ts_ + time_offset_gyro_to_cam;
        if (gyr_t < start_timestamp)
            continue;
        if (gyr_t > end_timestamp)
            break;

        const double dt = imu_data[gyr_idx].ts_ - imu_data[gyr_idx-1].ts_;
        Vec6_t update_se3quat;
        update_se3quat.setZero(); // translation zero
        // rotation update
        update_se3quat.tail<3>() = (imu_data[gyr_idx-1].gyr_ - imu_bias) * dt;

        const ::g2o::SE3Quat new_rotation = ::g2o::SE3Quat(update_se3quat);
        //init_rotation = init_rotation * rotation_gyro_to_cam * new_rotation;
        ++new_gyr_idx;
    }
    integrated_rotation = init_rotation.rotation().toRotationMatrix();

    // return new gyro index to make the loop quicker and do not start at 0 all the time
    return new_gyr_idx;
}

}
}
