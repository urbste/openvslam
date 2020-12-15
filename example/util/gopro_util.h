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
#include "openvslam/config.h"

class gopro_input_telemetry {
public:
    gopro_input_telemetry();
    ~gopro_input_telemetry();

    void set_imu_config(const std::shared_ptr<openvslam::imu::config>& config);

    bool read_telemetry(const std::string& telemetry_json_file);

    std::vector<openvslam::gps::data> get_gps_data() const;
    std::vector<openvslam::imu::data> get_imu_data() const;
    openvslam::gps::config get_gps_config() const;
    openvslam::imu::config get_imu_config() const;

    void get_gps_data_at_time(const double timestamp,
                              openvslam::gps::data& interpolated_data);

    void get_imu_data_between_time(const double last_timestamp,
                                   const double current_timestamp,
                                   std::vector<openvslam::imu::data>& imu_data);
private:
    std::vector<openvslam::gps::data> gps_data_;
    std::vector<openvslam::imu::data> imu_data_;
    openvslam::imu::config imu_config_;
    openvslam::gps::config gps_config_;
};

#endif // EXAMPLE_UTIL_GOPRO_UTIL_H
