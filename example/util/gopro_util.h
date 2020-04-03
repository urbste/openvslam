// created by Steffen Urban April 2020, urbste@gmail.com
#ifndef EXAMPLE_UTIL_GOPRO_UTIL_H
#define EXAMPLE_UTIL_GOPRO_UTIL_H

#include <string>
#include <vector>

#include "openvslam/io/gopro_telemetry_reader.h"
#include "openvslam/gps/config.h"
#include "openvslam/gps/data.h"
#include "openvslam/imu/config.h"
#include "openvslam/imu/data.h"

class gopro_input_telemetry {
public:
    gopro_input_telemetry();
    ~gopro_input_telemetry();
    bool read_telemetry(const std::string& telemetry_json_file);

    std::vector<openvslam::gps::data> get_gps_data() const;
    openvslam::gps::config get_gps_config() const;

private:
    std::vector<openvslam::gps::data> gps_data_;
    std::vector<openvslam::imu::data> imu_data_;
    openvslam::imu::config imu_config_;
    openvslam::gps::config gps_config_;
};

#endif // EXAMPLE_UTIL_GOPRO_UTIL_H
