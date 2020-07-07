
#include "openvslam/match/sparse_feature_aligner.h"
#include "openvslam/camera/perspective.h"
#include "openvslam/camera/equirectangular.h"
#include "openvslam/camera/fisheye.h"
#include "openvslam/camera/radial_division.h"

#include <opencv2/core.hpp>

namespace openvslam {

namespace match {


bool project_to_image_distorted(
        const data::frame* curr_frame,
        const Mat44_t& world_to_cam,
        const Vec3_t& pt3,
        Vec2_t& projection) {
    float x_right;
    bool is_visible = false;
    switch (curr_frame->camera_->model_type_) {
    case camera::model_type_t::Perspective: {
        auto c = static_cast<camera::perspective*>(curr_frame->camera_);
        is_visible = c->reproject_to_image_distorted(world_to_cam.block<3,3>(0,0), world_to_cam.block<3,1>(0,3),
                              pt3, projection, x_right);
    } break;
    case camera::model_type_t::Fisheye: {
        auto c = static_cast<camera::fisheye*>(curr_frame->camera_);
        is_visible = c->reproject_to_image_distorted(world_to_cam.block<3,3>(0,0), world_to_cam.block<3,1>(0,3),
                              pt3, projection, x_right);
    }break;
    case camera::model_type_t::Equirectangular: {
        auto c = static_cast<camera::equirectangular*>(curr_frame->camera_);
        is_visible = c->reproject_to_image_distorted(world_to_cam.block<3,3>(0,0), world_to_cam.block<3,1>(0,3),
                              pt3, projection, x_right);
    }break;
    case camera::model_type_t::RadialDivision: {
        auto c = static_cast<camera::radial_division*>(curr_frame->camera_);
        is_visible = c->reproject_to_image_distorted(world_to_cam.block<3,3>(0,0), world_to_cam.block<3,1>(0,3),
                              pt3, projection, x_right);
    }break;
    }
    return is_visible;
}

// need undistortion first as we get the 3D point from this
// pt -> pt_undist -> bearing -> bearing*depth
void convert_keypt_to_bearing(
        const data::keyframe* curr_frame,
        const cv::KeyPoint& kp,
        Vec3_t& bearing) {
    switch (curr_frame->camera_->model_type_) {
    case camera::model_type_t::Perspective: {
        auto c = static_cast<camera::perspective*>(curr_frame->camera_);
        const cv::KeyPoint undist_kp = c->undistort_keypoint(kp);
        bearing = c->convert_keypoint_to_bearing(undist_kp);
    } break;
    case camera::model_type_t::Fisheye: {
        auto c = static_cast<camera::fisheye*>(curr_frame->camera_);
        const cv::KeyPoint undist_kp = c->undistort_keypoint(kp);
        bearing = c->convert_keypoint_to_bearing(undist_kp);
    }break;
    case camera::model_type_t::Equirectangular: {
        auto c = static_cast<camera::equirectangular*>(curr_frame->camera_);
        const cv::KeyPoint undist_kp = c->undistort_keypoint(kp);
        bearing = c->convert_keypoint_to_bearing(undist_kp);
    }break;
    case camera::model_type_t::RadialDivision: {
        auto c = static_cast<camera::radial_division*>(curr_frame->camera_);
        const cv::KeyPoint undist_kp = c->undistort_keypoint(kp);
        bearing = c->convert_keypoint_to_bearing(undist_kp);
    }break;
    }
}


} // namespace match
} // namespace openvslam

