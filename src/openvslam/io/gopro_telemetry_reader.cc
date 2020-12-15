// created by Steffen Urban April 2020, urbste@gmail.com

#include "openvslam/io/gopro_telemetry_reader.h"
#include "openvslam/type.h"
#include <fstream>
#include <iostream>
#include <spdlog/spdlog.h>

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
    gopro_imu_data.resize(gyro.size());
    for (unsigned int i = 0; i < gyro.size(); ++i) {
        Vec3_t accl_vec;
        accl_vec << accl[i]["value"][1],
                accl[i]["value"][2], accl[i]["value"][0];
        Vec3_t gyro_vec;
        gyro_vec << gyro[i]["value"][1],
                gyro[i]["value"][2], gyro[i]["value"][0];

        gopro_imu_data[i] = openvslam::imu::data(accl[i]["value"][1],
                accl[i]["value"][2], accl[i]["value"][0],
                gyro[i]["value"][1], gyro[i]["value"][2],
                gyro[i]["value"][0], gyro[i]["cts"]);
        gopro_imu_data[i].ts_ /= 1000.0;  // to seconds
    }
    // now set imu config
    const double imu_hz = 1.0 / (gopro_imu_data[1].ts_ - gopro_imu_data[0].ts_);
    //gopro_imu_config = openvslam::imu::config("gopro_imu",imu_hz, Mat44_t::Identity(),
    //                                          0.0,0.0,0.0,0.0);
    spdlog::debug("Loaded GoPro IMU Telemetry. Found "+
                  std::to_string(gopro_imu_data.size())+" datapoints.");
    spdlog::debug("IMU was running at approx. "+std::to_string(imu_hz)+" Hz.");

    gopro_gps_data.resize(gps5.size());
    Vec3_t llh_ref;
    bool set_llh_ref = false;
    for (unsigned int i = 0; i < gps5.size(); ++i) {
        gopro_gps_data[i] = openvslam::gps::data(
                gps5[i]["value"][0],  // longitude
                gps5[i]["value"][1],  // latitude
                gps5[i]["value"][2],  // height
                gps5[i]["precision"], // precision
                gps5[i]["fix"],       // fix
                gps5[i]["value"][3],  // speed 2d
                gps5[i]["value"][4],  // speed 3d
                gps5[i]["cts"]);      // timestamp
        gopro_gps_data[i].ts_ /= 1000.0; // to seconds
        if (gopro_gps_data[i].fix_ && !set_llh_ref) {
            llh_ref = gopro_gps_data[i].llh_;
            set_llh_ref = true;
        }
        gopro_gps_data[i].Set_ENUReferenceLLH(llh_ref); // set local enu reference to first gps reading
    }

    const double gps_hz = 1.0 / (gopro_gps_data[1].ts_ - gopro_gps_data[0].ts_);
    gopro_gps_config = openvslam::gps::config("gopro_gps",gps_hz, Mat44_t::Identity());

    spdlog::debug("Loaded GoPro GPS Telemetry. Found "+
                  std::to_string(gopro_gps_data.size())+" datapoints.");
    spdlog::debug("GPS was running at approx. "+std::to_string(gps_hz)+" Hz.");

    file.close();

    return true;
}

} // namespace io
} // namespace openvslam
