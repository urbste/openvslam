#include "openvslam/mapping_module.h"
#include "openvslam/data/keyframe.h"
#include "openvslam/data/landmark.h"
#include "openvslam/data/map_database.h"
#include "openvslam/module/gps_initializer.h"
#include "openvslam/gps/data.h"

#include <spdlog/spdlog.h>
#include <iostream>
namespace openvslam {
namespace module {

gps_initializer::gps_initializer(data::map_database* map_db)
    : map_db_(map_db) {
    spdlog::debug("CONSTRUCT: module::gps_initializer");
}

void gps_initializer::set_mapping_module(mapping_module* mapper) {
    mapper_ = mapper;
}

void gps_initializer::enable_gps_initializer() {
    is_initializer_enabled_ = true;
}

void gps_initializer::disable_gps_initializer() {
    is_initializer_enabled_ = false;
}

bool gps_initializer::is_enabled() const {
    return is_initializer_enabled_;
}

void mean_of_eigen_vec(const eigen_alloc_vector<Vec3_t>& in_vec,
                       Vec3_t& mean) {
    mean.setZero();
    for (int i = 0; i < in_vec.size(); ++i) {
        for (int j = 0; j < 3; ++j) {
            mean[j] += in_vec[i][j];
        }
    }
    const double nr_els = static_cast<double>(in_vec.size());
    for (int j = 0; j < 3; ++j) {
        mean[j] /= nr_els;
    }
}

bool gps_initializer::start_map_scale_initalization() {

    if (!is_initializer_enabled_) {
        return false;
    }
    // loop all keyframes and start it
    auto kfs = map_db_->get_all_keyframes();
    eigen_alloc_vector<Vec3_t> gps_pos;
    std::vector<double> sigma;
    eigen_alloc_vector<Vec3_t> cam_pos;

    for (auto kf : kfs) {
        const auto gps = kf->get_gps_data();
        if (gps.fix_ == gps::gps_fix_state_t::FIX_3D) {
            gps_pos.push_back(gps.xyz_);
            cam_pos.push_back(kf->get_cam_center());
        }
    }
    double total_distance_traveled = 0.0;
    // calculate traveled distance
    //for (size_t i = 1; i < gps_pos.size(); ++i) {
    total_distance_traveled = (gps_pos[gps_pos.size()-1]-gps_pos[0]).norm();
    //}

    if (total_distance_traveled < min_traveled_distance_) {
        state_ = gps_initializer_state_t::AwaitingScaleInit;
        return false;
    }

    // now calculate scale
    Vec3_t mean_gps;
    Vec3_t mean_cam;
    mean_of_eigen_vec(gps_pos, mean_gps);
    mean_of_eigen_vec(cam_pos, mean_cam);

    double sum_gps_diff = 0.0;
    double sum_cam_diff = 0.0;
    for (size_t i = 0; i < gps_pos.size(); ++i) {
        sum_gps_diff += (gps_pos[i] - mean_gps).squaredNorm();
        sum_cam_diff += (cam_pos[i] - mean_cam).squaredNorm();
    }
    double scale = std::sqrt(sum_gps_diff/sum_cam_diff);

    // scale map
    // stop all threads and scale the map
    {
        std::lock_guard<std::mutex> lock(mtx_thread_);
        gps_scaling_is_running_ = true;
        abort_gps_scaling_ = false;
    }
    // stop mapping module
    mapper_->request_pause();
    while (!mapper_->is_paused() && !mapper_->is_terminated()) {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    // lock the map
    std::lock_guard<std::mutex> lock2(data::map_database::mtx_database_);

    auto landmarks = map_db_->get_all_landmarks();
    for (auto lm : landmarks) {
        if (!lm) {
            continue;
        }
        lm->set_pos_in_world(lm->get_pos_in_world() * scale);
    }
    for (auto kf : kfs) {
        Mat44_t cam_pose_cw = kf->get_cam_pose();
        cam_pose_cw.block<3, 1>(0, 3) *= scale;
        kf->set_cam_pose(cam_pose_cw);
    }
    state_ = gps_initializer_state_t::ScaleInitSucceeded;
    // finished scaling return to normal behaviour
    mapper_->resume();

    gps_scaling_is_running_ = false;

    spdlog::info("updated the map");
}

bool gps_initializer::start_map_rotation_initalization() {}

void gps_initializer::abort() {
    std::lock_guard<std::mutex> lock(mtx_thread_);
    abort_gps_scaling_ = true;
}

bool gps_initializer::is_running() const {
    std::lock_guard<std::mutex> lock(mtx_thread_);
    return gps_scaling_is_running_;
}

}
}
