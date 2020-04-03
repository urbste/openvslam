// created by Steffen Urban April 2020, urbste@gmail.com

#include "openvslam/io/gopro_telemetry_reader.h"
#include "openvslam/type.h"
#include <fstream>

namespace openvslam {
namespace io {

using json = nlohmann::json;
bool ReadGoProTelemetryJson(const std::string& path_to_telemetry_file,
                            openvslam::gps::config& gopro_gps_config,
                            openvslam::imu::config& gopro_imu_config,
                            std::vector<openvslam::gps::data>& gopro_gps_data,
                            std::vector<openvslam::imu::data>& gopro_imu_data) {
    std::ifstream file;
    file.open(path_to_telemetry_file.c_str());
    json j;
    file >> j;

    const auto accl = j["1"]["streams"]["ACCL"]["samples"];
    const auto gyro = j["1"]["streams"]["GYRO"]["samples"];
    const auto gps5 = j["1"]["streams"]["GPS5"]["samples"];

    for (const auto& e : accl) {
        Vec3_t v;
        v << e["value"][0], e["value"][1], e["value"][2];
        const double timestamp = e["cts"];
    }

    for (const auto& e : gyro) {
        Vec3_t v;
        v << e["value"][0], e["value"][1], e["value"][2];
        const double timestamp = e["cts"];
    }
    // now set imu config
    //const double imu_hz = 1000.0 /
    //gopro_gps_config = openvslam::gps::config("gopro_imu",hz, Mat44_t::Identity());

    for (const auto& e : gps5) {
        Vec3_t lle;
        Vec3_t vel2d_vel3d;
        lle << e["value"][0], e["value"][1], e["value"][2];
        vel2d_vel3d << e["value"][3], e["value"][4];
        const double timestamp = e["cts"];
        const double dop = e["precision"];
    }
    //const double gps_hz = 1000.0/
    //gopro_gps_config = openvslam::gps::config("gopro_gps",gps_hz, Mat44_t::Identity());


    file.close();

    return true;
}

} // namespace io
} // namespace openvslam
