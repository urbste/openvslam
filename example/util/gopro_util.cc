// created by Steffen Urban April 2020, urbste@gmail.com
#include "gopro_util.h"
#include "openvslam/type.h"

#include <string>
#include <vector>
#include <iostream>

gopro_input_telemetry::gopro_input_telemetry() {}
gopro_input_telemetry::~gopro_input_telemetry() {}

bool gopro_input_telemetry::read_telemetry(const std::string& telemetry_json_file) {
    return openvslam::io::ReadGoProTelemetryJson(telemetry_json_file,
                                                 gps_config_,
                                                 imu_config_,
                                                 gps_data_,
                                                 imu_data_);
}

// linear interpolation, t \el [0,1]
double lerp(const double a,const  double b,const  double t) {
    return a + t * (b - a);
}

void interpolate_gps_data_for_cam_time(const std::vector<openvslam::gps::data>& gps_data,
                                    const double image_timestamp, const double gps_frequency_ms,
                                    openvslam::gps::data& interpolated_gps_data) {

    const double last_gps_timestamp = gps_data[gps_data.size()-1].ts_;

    // gps_data is sorted from smallest to largest timestamp
    for (size_t data_idx = 0; data_idx < gps_data.size() - 1; ++data_idx) {
        if (image_timestamp > last_gps_timestamp)
            break;

        const size_t data_idx_1 = data_idx + 1;

        const double delta_t1 = image_timestamp - gps_data[data_idx].ts_;
        if (delta_t1 < 0)
            continue;

        const double delta_t2 = image_timestamp - gps_data[data_idx_1].ts_;
        if (delta_t2 <= 0 && delta_t1 >= 0) {

            const double delta_t1_abs = std::abs(delta_t1);
            // now interpolate linearly between two gps timestamps
            const double gps_t1 = gps_data[data_idx].ts_;
            const double gps_t2 = gps_data[data_idx_1].ts_;
            const double diff_gps_t = gps_t2 - gps_t1;
            const double diff_cam_gps_t1 = delta_t1_abs / diff_gps_t; // fraction from 0->1

            openvslam::Vec3_t xyz;
            for (int v = 0; v < 3; ++v) {
                xyz[v] = lerp(gps_data[data_idx].xyz_[v],gps_data[data_idx_1].xyz_[v], diff_cam_gps_t1);
            }
            interpolated_gps_data.Set_XYZ(xyz);
            interpolated_gps_data.ts_ = gps_t1 + delta_t1_abs;
            // this is probably a wrong assumption but will try this for now
            interpolated_gps_data.dop_precision_ =
                    lerp(gps_data[data_idx].dop_precision_,
                         gps_data[data_idx_1].dop_precision_, diff_cam_gps_t1);
            interpolated_gps_data.speed_3d_ =
                    lerp(gps_data[data_idx].speed_3d_,gps_data[data_idx_1].speed_3d_, diff_cam_gps_t1);
            interpolated_gps_data.speed_2d_ =
                    lerp(gps_data[data_idx].speed_2d_,gps_data[data_idx_1].speed_2d_, diff_cam_gps_t1);
            break;
        }
    }
}

void gopro_input_telemetry::get_gps_data_at_time(const double camera_timestamp,
                                          openvslam::gps::data& interpolated_data) {

    if (camera_timestamp > gps_data_[gps_data_.size()-1].ts_) {
        interpolated_data = gps_data_[gps_data_.size()-1];
    } else {
        interpolate_gps_data_for_cam_time(gps_data_, camera_timestamp,
                                          gps_config_.get_rate_dt(),interpolated_data);
    }
}
