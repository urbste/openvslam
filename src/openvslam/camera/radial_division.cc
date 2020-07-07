// Created by Steffen Urban June 2019, urbste@googlemail.com, github.com/urbste

#include "openvslam/camera/radial_division.h"

#include <iostream>

#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>

namespace openvslam {
namespace camera {

radial_division::radial_division(const std::string& name, const setup_type_t& setup_type, const color_order_t& color_order,
                                 const unsigned int cols, const unsigned int rows, const double fps,
                                 const double fx, const double fy, const double cx, const double cy,
                                 const double distortion, const double focal_x_baseline, const double resize_fac)
    : base(name, setup_type, model_type_t::RadialDivision, color_order, cols, rows, fps,
           focal_x_baseline, focal_x_baseline / fx, 64*resize_fac, 48*resize_fac, resize_fac),
      fx_(fx * resize_fac), fy_(fy * resize_fac), cx_(cx * resize_fac), cy_(cy * resize_fac),
      fx_inv_(1.0 / (fx * resize_fac)), fy_inv_(1.0 / (fy * resize_fac)),
      distortion_(distortion), resize_fac_(resize_fac) {
    spdlog::debug("CONSTRUCT: camera::radial_division");

    cv_cam_matrix_ = (cv::Mat_<float>(3, 3) << fx_, 0, cx_, 0, fy_, cy_, 0, 0, 1);

    eigen_cam_matrix_ << fx_, 0, cx_, 0, fy_, cy_, 0, 0, 1;

    img_bounds_ = compute_image_bounds();

    inv_cell_width_ = static_cast<double>(num_grid_cols_) / (img_bounds_.max_x_ - img_bounds_.min_x_);
    inv_cell_height_ = static_cast<double>(num_grid_rows_) / (img_bounds_.max_y_ - img_bounds_.min_y_);
}

radial_division::radial_division(const YAML::Node& yaml_node)
    : radial_division(yaml_node["Camera.name"].as<std::string>(),
                      load_setup_type(yaml_node),
                      load_color_order(yaml_node),
                      yaml_node["Camera.cols"].as<unsigned int>(),
                      yaml_node["Camera.rows"].as<unsigned int>(),
                      yaml_node["Camera.fps"].as<double>(),
                      yaml_node["Camera.fx"].as<double>(),
                      yaml_node["Camera.fy"].as<double>(),
                      yaml_node["Camera.cx"].as<double>(),
                      yaml_node["Camera.cy"].as<double>(),
                      yaml_node["Camera.distortion"].as<double>(),
                      yaml_node["Camera.focal_x_baseline"].as<double>(0.0),
                      yaml_node["Camera.resize_fac"].as<double>(0.0)) {}

radial_division::~radial_division() {
    spdlog::debug("DESTRUCT: camera::radial_division");
}

void radial_division::show_parameters() const {
    show_common_parameters();
    std::cout << "  - fx: " << fx_ << std::endl;
    std::cout << "  - fy: " << fy_ << std::endl;
    std::cout << "  - cx: " << cx_ << std::endl;
    std::cout << "  - cy: " << cy_ << std::endl;
    std::cout << "  - distortion: " << distortion_ << std::endl;
    std::cout << "  - min x: " << img_bounds_.min_x_ << std::endl;
    std::cout << "  - max x: " << img_bounds_.max_x_ << std::endl;
    std::cout << "  - min y: " << img_bounds_.min_y_ << std::endl;
    std::cout << "  - max y: " << img_bounds_.max_y_ << std::endl;
}

image_bounds radial_division::compute_image_bounds() const {
    spdlog::debug("compute image bounds");

    if (distortion_ == 0.0) {
        return image_bounds{0.0, cols_, 0.0, rows_};
    }
    else {
        const std::vector<cv::KeyPoint> corners{cv::KeyPoint(0.0, 0.0, 1.0),
                                                cv::KeyPoint(cols_, 0.0, 1.0),
                                                cv::KeyPoint(0.0, rows_, 1.0),
                                                cv::KeyPoint(cols_, rows_, 1.0)};

        std::vector<cv::KeyPoint> undist_corners;
        undistort_keypoints(corners, undist_corners);

        return image_bounds{std::min(undist_corners.at(0).pt.x, undist_corners.at(2).pt.x),
                            std::max(undist_corners.at(1).pt.x, undist_corners.at(3).pt.x),
                            std::min(undist_corners.at(0).pt.y, undist_corners.at(1).pt.y),
                            std::max(undist_corners.at(2).pt.y, undist_corners.at(3).pt.y)};
    }
}

void radial_division::undistort_keypoints(const std::vector<cv::KeyPoint>& dist_keypts, std::vector<cv::KeyPoint>& undist_keypts) const {
    // fill cv::Mat with distorted keypoints
    undist_keypts.resize(dist_keypts.size());
    for (unsigned long idx = 0; idx < dist_keypts.size(); ++idx) {
        undist_keypts.at(idx) = this->undistort_keypoint(dist_keypts.at(idx));
        undist_keypts.at(idx).angle = dist_keypts.at(idx).angle;
        undist_keypts.at(idx).size = dist_keypts.at(idx).size;
        undist_keypts.at(idx).octave = dist_keypts.at(idx).octave;
    }
}

void radial_division::convert_keypoints_to_bearings(const std::vector<cv::KeyPoint>& undist_keypts, eigen_alloc_vector<Vec3_t>& bearings) const {
    bearings.resize(undist_keypts.size());
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for (unsigned long idx = 0; idx < undist_keypts.size(); ++idx) {
        bearings.at(idx) = this->convert_keypoint_to_bearing(undist_keypts.at(idx));
    }
}

void radial_division::convert_bearings_to_keypoints(const eigen_alloc_vector<Vec3_t>& bearings, std::vector<cv::KeyPoint>& undist_keypts) const {
    undist_keypts.resize(bearings.size());
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for (unsigned long idx = 0; idx < bearings.size(); ++idx) {
        undist_keypts.at(idx) = convert_bearing_to_keypoint(bearings.at(idx));
    }
}

bool radial_division::reproject_to_image(const Mat33_t& rot_cw, const Vec3_t& trans_cw, const Vec3_t& pos_w, Vec2_t& reproj, float& x_right) const {
    const Vec3_t pos_c = rot_cw * pos_w + trans_cw;

    if (pos_c(2) <= 0.0) {
        return false;
    }

    const auto z_inv = 1.0 / pos_c(2);
    reproj(0) = fx_ * pos_c(0) * z_inv + cx_;
    reproj(1) = fy_ * pos_c(1) * z_inv + cy_;
    x_right = reproj(0) - focal_x_baseline_ * z_inv;

    if (reproj(0) < img_bounds_.min_x_ || reproj(0) > img_bounds_.max_x_) {
        return false;
    }
    if (reproj(1) < img_bounds_.min_y_ || reproj(1) > img_bounds_.max_y_) {
        return false;
    }

    return true;
}

bool radial_division::reproject_to_image_distorted(const Mat33_t& rot_cw, const Vec3_t& trans_cw, const Vec3_t& pos_w, Vec2_t& reproj, float& x_right) const {

    // convert to camera-coordinates
    const Vec3_t pos_c = rot_cw * pos_w + trans_cw;

    // check if the point is visible
    if (pos_c(2) <= std::numeric_limits<double>::epsilon()) {
        return false;
    }

    // do distorted projection
    const double xs = pos_c[0] / pos_c[2];
    const double ys = pos_c[1] / pos_c[2];
    const double r2 = (xs * xs + ys * ys) * distortion_;

    const double denom = 2.0 * r2;
    const double inner = 1.0 - 4.0 * r2;

    if (std::abs(denom) < std::numeric_limits<double>::epsilon() || inner < 0.0) {
        reproj(0) = fx_ * xs + cx_;
        reproj(1) = fy_ * ys + cy_;
    } else {
        const double scale = (1.0 - std::sqrt(inner)) / denom;
        reproj(0) = xs * fx_ * scale + cx_;
        reproj(1) = ys * fy_ * scale + cy_;
    }

    // reproject onto the image
    x_right = reproj(0) - focal_x_baseline_ / pos_c[2];

    // check if the point is visible
    return (0.0 < reproj(0) && reproj(0) < static_cast<double>(cols_)
            && 0.0 < reproj(1) && reproj(1) < static_cast<double>(rows_));
}

bool radial_division::reproject_to_bearing(const Mat33_t& rot_cw, const Vec3_t& trans_cw, const Vec3_t& pos_w, Vec3_t& reproj) const {
    reproj = rot_cw * pos_w + trans_cw;

    if (reproj(2) <= 0.0) {
        return false;
    }

    const auto z_inv = 1.0 / reproj(2);
    const auto x = fx_ * reproj(0) * z_inv + cx_;
    const auto y = fy_ * reproj(1) * z_inv + cy_;

    if (x < img_bounds_.min_x_ || x > img_bounds_.max_x_) {
        return false;
    }
    if (y < img_bounds_.min_y_ || y > img_bounds_.max_y_) {
        return false;
    }

    reproj.normalize();

    return true;
}

nlohmann::json radial_division::to_json() const {
    return {
        {"model_type", get_model_type_string()},
        {"setup_type", get_setup_type_string()},
        {"color_order", get_color_order_string()},
        {"cols", cols_},
        {"rows", rows_},
        {"fps", fps_},
        {"focal_x_baseline", focal_x_baseline_},
        {"num_grid_cols", num_grid_cols_},
        {"num_grid_rows", num_grid_rows_},
        {"fx", fx_},
        {"fy", fy_},
        {"cx", cx_},
        {"cy", cy_},
        {"distortion", distortion_},
    };
}
void radial_division::jacobian_xyz_to_cam(const Vec3_t &xyz,
                                      Mat26_t &jac, const double scale) const {

    // using Matlab auto generated jacs, they are just sooo much faster than cv::projectPoints jacobians
    const double rx = 0.0;
    const double ry = 0.0;
    const double rz = 0.0;

    const double tx = 0.0;
    const double ty = 0.0;
    const double tz = 0.0;

    const double X = xyz[0];
    const double Y = xyz[1];
    const double Z = xyz[2];

    const double t7 = Y*rz;
    const double t8 = Z*ry;
    const double t2 = X-t7+t8+tx;
    const double t12 = X*rz;
    const double t13 = Z*rx;
    const double t3 = Y+t12-t13+ty;
    const double t4 = Y*rx;
    const double t10 = X*ry;
    const double t5 = Z+t4-t10+tz;
    const double t6 = 1.0/(t5*t5);
    const double t9 = t2*t2;
    const double t11 = t6*t9;
    const double t14 = t3*t3;
    const double t15 = t6*t14;
    const double t16 = t11+t15;
    const double t17 = 1.0/t16;
    const double t20 = distortion_*t16*4.0;
    const double t18 = -t20+1.0;
    const double t19 = 1.0/distortion_;
    const double t21 = sqrt(t18);
    const double t22 = t21-1.0;
    const double t23 = X*2.0;
    const double t24 = tx*2.0;
    const double t25 = Z*ry*2.0;
    const double t51 = Y*rz*2.0;
    const double t26 = t23+t24+t25-t51;
    const double t27 = 1.0/(t5*t5*t5);
    const double t28 = 1.0/sqrt(t18);
    const double t29 = 1.0/(t16*t16);
    const double t30 = Y*2.0;
    const double t31 = ty*2.0;
    const double t32 = X*rz*2.0;
    const double t52 = Z*rx*2.0;
    const double t33 = t30+t31+t32-t52;
    const double t34 = X*X;
    const double t35 = Y*Y;
    const double t36 = rz*rz;
    const double t37 = Z*Z;
    const double t38 = 1.0/t5;
    const double t39 = Z*t3*t6*2.0;
    const double t40 = Y*t9*t27*2.0;
    const double t41 = Y*t14*t27*2.0;
    const double t42 = t39+t40+t41;
    const double t43 = Z*t2*t6*2.0;
    const double t44 = X*t9*t27*2.0;
    const double t45 = X*t14*t27*2.0;
    const double t46 = t43+t44+t45;
    const double t47 = X*ty;
    const double t48 = rz*t34;
    const double t49 = rz*t35;
    const double t68 = Y*tx;
    const double t69 = X*Z*rx;
    const double t70 = Y*Z*ry;
    const double t50 = t47+t48+t49-t68-t69-t70;
    const double t53 = X*tx*2.0;
    const double t54 = Y*ty*2.0;
    const double t55 = tx*tx;
    const double t56 = ty*ty;
    const double t57 = t34*t36;
    const double t58 = t35*t36;
    const double t59 = rx*rx;
    const double t60 = t37*t59;
    const double t61 = ry*ry;
    const double t62 = t37*t61;
    const double t63 = X*Z*ry*2.0;
    const double t64 = X*rz*ty*2.0;
    const double t65 = Z*ry*tx*2.0;
    const double t66 = t34+t35+t53+t54+t55+t56+t57+t58+t60+t62+t63+t64+t65-Y*Z*rx*2.0-Y*rz*tx*2.0-Z*rx*ty*2.0-X*Z*rx*rz*2.0-Y*Z*ry*rz*2.0;
    const double t67 = 1.0/t66;

    const double fx_scaled = scale * fx_;
    const double fy_scaled = scale * fy_;
    jac.setZero();
    jac(0,0) = fx_scaled*t17*t19*t22*t38*(-1.0/2.0)+fx_scaled*t2*t17*t26*t27*t28+fx_scaled*t2*t19*t22*t26*t27*t29*(1.0/2.0);
    jac(0,1) = fx_scaled*t2*t17*t27*t28*t33+fx_scaled*t2*t19*t22*t27*t29*t33*(1.0/2.0);
    jac(0,2) = fx_scaled*t2*t19*t22*t28*t67*(1.0/2.0);
    jac(0,3) = -fx_scaled*t2*t17*t28*t38*t42+Y*fx_scaled*t2*t6*t17*t19*t22*(1.0/2.0)-fx_scaled*t2*t19*t22*t29*t38*t42*(1.0/2.0);
    jac(0,4) = fx_scaled*t2*t17*t28*t38*t46-Z*fx_scaled*t17*t19*t22*t38*(1.0/2.0)-X*fx_scaled*t2*t6*t17*t19*t22*(1.0/2.0)+fx_scaled*t2*t19*t22*t29*t38*t46*(1.0/2.0);
    jac(0,5) = fx_scaled*t2*t17*t27*t28*t50*2.0+Y*fx_scaled*t17*t19*t22*t38*(1.0/2.0)+fx_scaled*t2*t19*t22*t27*t29*t50;
    jac(1,0) = fy_scaled*t3*t17*t26*t27*t28+fy_scaled*t3*t19*t22*t26*t27*t29*(1.0/2.0);
    jac(1,1) = fy_scaled*t17*t19*t22*t38*(-1.0/2.0)+fy_scaled*t3*t17*t27*t28*t33+fy_scaled*t3*t19*t22*t27*t29*t33*(1.0/2.0);
    jac(1,2) = fy_scaled*t3*t19*t22*t28*t67*(1.0/2.0);
    jac(1,3) = -fy_scaled*t3*t17*t28*t38*t42+Z*fy_scaled*t17*t19*t22*t38*(1.0/2.0)+Y*fy_scaled*t3*t6*t17*t19*t22*(1.0/2.0)-fy_scaled*t3*t19*t22*t29*t38*t42*(1.0/2.0);
    jac(1,4) = fy_scaled*t3*t17*t28*t38*t46-X*fy_scaled*t3*t6*t17*t19*t22*(1.0/2.0)+fy_scaled*t3*t19*t22*t29*t38*t46*(1.0/2.0);
    jac(1,5) = fy_scaled*t3*t17*t27*t28*t50*2.0-X*fy_scaled*t17*t19*t22*t38*(1.0/2.0)+fy_scaled*t3*t19*t22*t27*t29*t50;
    jac *= -1.0;

}

} // namespace camera
} // namespace openvslam
