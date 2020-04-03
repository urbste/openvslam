// created by Steffen Urban April 2020, urbste@gmail.com
#ifndef OPENVSLAM_GOPRO_TELEMETRY_READER_H
#define OPENVSLAM_GOPRO_TELEMETRY_READER_H

#include "openvslam/gps/config.h"
#include "openvslam/gps/data.h"
#include "openvslam/imu/config.h"
#include "openvslam/imu/data.h"
#include <nlohmann/json.hpp>
#include <string>

namespace openvslam {
namespace io {

bool ReadGoProTelemetryJson(const std::string& path_to_telemetry_file,
                            openvslam::gps::config& gopro_gps_config,
                            openvslam::imu::config& gopro_imu_config,
                            std::vector<openvslam::gps::data>& gopro_gps_data,
                            std::vector<openvslam::imu::data>& gopro_imu_data);

} // namespace io
} // namespace openvslam

#endif // OPENVSLAM_GOPRO_TELEMETRY_READER_H
