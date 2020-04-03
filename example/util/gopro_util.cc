// created by Steffen Urban April 2020, urbste@gmail.com
#include "gopro_util.h"

#include <string>
#include <vector>

gopro_input_telemetry::gopro_input_telemetry() {}
gopro_input_telemetry::~gopro_input_telemetry() {}

bool gopro_input_telemetry::read_telemetry(const std::string& telemetry_json_file) {
    return openvslam::io::ReadGoProTelemetryJson(telemetry_json_file,
                                                 gps_config_,
                                                 imu_config_,
                                                 gps_data_,
                                                 imu_data_);
}
