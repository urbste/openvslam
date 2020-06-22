#include "openvslam/camera/perspective.h"

#include <iostream>

#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>
#include <chrono>
using time_now = std::chrono::steady_clock;

namespace openvslam {
namespace camera {

perspective::perspective(const std::string& name, const setup_type_t& setup_type, const color_order_t& color_order,
                         const unsigned int cols, const unsigned int rows, const double fps,
                         const double fx, const double fy, const double cx, const double cy,
                         const double k1, const double k2, const double p1, const double p2, const double k3,
                         const double focal_x_baseline)
    : base(name, setup_type, model_type_t::Perspective, color_order, cols, rows, fps, focal_x_baseline, focal_x_baseline / fx),
      fx_(fx), fy_(fy), cx_(cx), cy_(cy), fx_inv_(1.0 / fx), fy_inv_(1.0 / fy),
      k1_(k1), k2_(k2), p1_(p1), p2_(p2), k3_(k3) {
    spdlog::debug("CONSTRUCT: camera::perspective");

    cv_cam_matrix_ = (cv::Mat_<double>(3, 3) << fx_, 0, cx_, 0, fy_, cy_, 0, 0, 1);
    cv_dist_params_ = (cv::Mat_<double>(5, 1) << k1_, k2_, p1_, p2_, k3_);

    eigen_cam_matrix_ << fx_, 0, cx_, 0, fy_, cy_, 0, 0, 1;
    eigen_dist_params_ << k1_, k2_, p1_, p2_, k3_;

    img_bounds_ = compute_image_bounds();

    inv_cell_width_ = static_cast<double>(num_grid_cols_) / (img_bounds_.max_x_ - img_bounds_.min_x_);
    inv_cell_height_ = static_cast<double>(num_grid_rows_) / (img_bounds_.max_y_ - img_bounds_.min_y_);
}

perspective::perspective(const YAML::Node& yaml_node)
    : perspective(yaml_node["Camera.name"].as<std::string>(),
                  load_setup_type(yaml_node),
                  load_color_order(yaml_node),
                  yaml_node["Camera.cols"].as<unsigned int>(),
                  yaml_node["Camera.rows"].as<unsigned int>(),
                  yaml_node["Camera.fps"].as<double>(),
                  yaml_node["Camera.fx"].as<double>(),
                  yaml_node["Camera.fy"].as<double>(),
                  yaml_node["Camera.cx"].as<double>(),
                  yaml_node["Camera.cy"].as<double>(),
                  yaml_node["Camera.k1"].as<double>(),
                  yaml_node["Camera.k2"].as<double>(),
                  yaml_node["Camera.p1"].as<double>(),
                  yaml_node["Camera.p2"].as<double>(),
                  yaml_node["Camera.k3"].as<double>(),
                  yaml_node["Camera.focal_x_baseline"].as<double>(0.0)) {}

perspective::~perspective() {
    spdlog::debug("DESTRUCT: camera::perspective");
}

void perspective::show_parameters() const {
    show_common_parameters();
    std::cout << "  - fx: " << fx_ << std::endl;
    std::cout << "  - fy: " << fy_ << std::endl;
    std::cout << "  - cx: " << cx_ << std::endl;
    std::cout << "  - cy: " << cy_ << std::endl;
    std::cout << "  - k1: " << k1_ << std::endl;
    std::cout << "  - k2: " << k2_ << std::endl;
    std::cout << "  - p1: " << p1_ << std::endl;
    std::cout << "  - p2: " << p2_ << std::endl;
    std::cout << "  - k3: " << k3_ << std::endl;
    std::cout << "  - min x: " << img_bounds_.min_x_ << std::endl;
    std::cout << "  - max x: " << img_bounds_.max_x_ << std::endl;
    std::cout << "  - min y: " << img_bounds_.min_y_ << std::endl;
    std::cout << "  - max y: " << img_bounds_.max_y_ << std::endl;
}

image_bounds perspective::compute_image_bounds() const {
    spdlog::debug("compute image bounds");

    if (k1_ == 0 && k2_ == 0 && p1_ == 0 && p2_ == 0 && k3_ == 0) {
        // any distortion does not exist

        return image_bounds{0.0, cols_, 0.0, rows_};
    }
    else {
        // distortion exists

        // corner coordinates: (x, y) = (col, row)
        const std::vector<cv::KeyPoint> corners{cv::KeyPoint(0.0, 0.0, 1.0),      // left top
                                                cv::KeyPoint(cols_, 0.0, 1.0),    // right top
                                                cv::KeyPoint(0.0, rows_, 1.0),    // left bottom
                                                cv::KeyPoint(cols_, rows_, 1.0)}; // right bottom

        std::vector<cv::KeyPoint> undist_corners;
        undistort_keypoints(corners, undist_corners);

        return image_bounds{std::min(undist_corners.at(0).pt.x, undist_corners.at(2).pt.x),
                            std::max(undist_corners.at(1).pt.x, undist_corners.at(3).pt.x),
                            std::min(undist_corners.at(0).pt.y, undist_corners.at(1).pt.y),
                            std::max(undist_corners.at(2).pt.y, undist_corners.at(3).pt.y)};
    }
}

void perspective::undistort_keypoints(const std::vector<cv::KeyPoint>& dist_keypts, std::vector<cv::KeyPoint>& undist_keypts) const {
    // cv::undistortPoints does not accept an empty input
    if (dist_keypts.empty()) {
        undist_keypts.clear();
        return;
    }

    // fill cv::Mat with distorted keypoints
    cv::Mat mat(dist_keypts.size(), 2, CV_32F);
    for (unsigned long idx = 0; idx < dist_keypts.size(); ++idx) {
        mat.at<float>(idx, 0) = dist_keypts.at(idx).pt.x;
        mat.at<float>(idx, 1) = dist_keypts.at(idx).pt.y;
    }

    // undistort
    mat = mat.reshape(2);
    cv::undistortPoints(mat, mat, cv_cam_matrix_, cv_dist_params_, cv::Mat(), cv_cam_matrix_,
                        cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 20, 1e-6));
    mat = mat.reshape(1);

    // convert to cv::Mat
    undist_keypts.resize(dist_keypts.size());
    for (unsigned long idx = 0; idx < undist_keypts.size(); ++idx) {
        undist_keypts.at(idx).pt.x = mat.at<float>(idx, 0);
        undist_keypts.at(idx).pt.y = mat.at<float>(idx, 1);
        undist_keypts.at(idx).angle = dist_keypts.at(idx).angle;
        undist_keypts.at(idx).size = dist_keypts.at(idx).size;
        undist_keypts.at(idx).octave = dist_keypts.at(idx).octave;
    }
}

void perspective::convert_keypoints_to_bearings(const std::vector<cv::KeyPoint>& undist_keypts, eigen_alloc_vector<Vec3_t>& bearings) const {
    bearings.resize(undist_keypts.size());
    for (unsigned long idx = 0; idx < undist_keypts.size(); ++idx) {
        const auto x_normalized = (undist_keypts.at(idx).pt.x - cx_) / fx_;
        const auto y_normalized = (undist_keypts.at(idx).pt.y - cy_) / fy_;
        const auto l2_norm = std::sqrt(x_normalized * x_normalized + y_normalized * y_normalized + 1.0);
        bearings.at(idx) = Vec3_t{x_normalized / l2_norm, y_normalized / l2_norm, 1.0 / l2_norm};
    }
}

void perspective::convert_bearings_to_keypoints(const eigen_alloc_vector<Vec3_t>& bearings, std::vector<cv::KeyPoint>& undist_keypts) const {
    undist_keypts.resize(bearings.size());
    for (unsigned long idx = 0; idx < bearings.size(); ++idx) {
        const auto x_normalized = bearings.at(idx)(0) / bearings.at(idx)(2);
        const auto y_normalized = bearings.at(idx)(1) / bearings.at(idx)(2);

        undist_keypts.at(idx).pt.x = fx_ * x_normalized + cx_;
        undist_keypts.at(idx).pt.y = fy_ * y_normalized + cy_;
    }
}

bool perspective::reproject_to_image(const Mat33_t& rot_cw, const Vec3_t& trans_cw, const Vec3_t& pos_w, Vec2_t& reproj, float& x_right) const {
    // convert to camera-coordinates
    const Vec3_t pos_c = rot_cw * pos_w + trans_cw;

    // check if the point is visible
    if (pos_c(2) <= 0.0) {
        return false;
    }

    // reproject onto the image
    const auto z_inv = 1.0 / pos_c(2);
    reproj(0) = fx_ * pos_c(0) * z_inv + cx_;
    reproj(1) = fy_ * pos_c(1) * z_inv + cy_;
    x_right = reproj(0) - focal_x_baseline_ * z_inv;

    // check if the point is visible
    return (img_bounds_.min_x_ < reproj(0) && reproj(0) < img_bounds_.max_x_
            && img_bounds_.min_y_ < reproj(1) && reproj(1) < img_bounds_.max_y_);
}

bool perspective::reproject_to_image_distorted(const Mat33_t& rot_cw, const Vec3_t& trans_cw, const Vec3_t& pos_w, Vec2_t& reproj, float& x_right) const {

    // convert to camera-coordinates
    const Vec3_t pos_c = rot_cw * pos_w + trans_cw;

    // check if the point is visible
    if (pos_c(2) <= 0.0) {
        return false;
    }

    // do distorted projection
    const double xs = pos_c[0] / pos_c[2];
    const double ys = pos_c[1] / pos_c[2];
    const double xsys = 2.0* xs * ys;
    const double r2 = xs * xs + ys * ys;
    const double r4 = r2 * r2;
    const double r6 = r4*r2;
    const double dist = 1.0 + k1_*r2 + k2_*r4 + k3_*r6;
    reproj(0) = fx_ * (xs * dist + p1_ * xsys + p2_ * (r2 + 2.0 * xs * xs)) + cx_;
    reproj(1) = fy_ * (ys * dist + p1_ * (r2 + 2.0 * ys * ys) + p2_ * xsys) + cy_;

    // reproject onto the image
    x_right = reproj(0) - focal_x_baseline_ / pos_c[2];

    // check if the point is visible
    return (0.0 < reproj(0) && reproj(0) < static_cast<double>(cols_)
            && 0.0 < reproj(1) && reproj(1) < static_cast<double>(rows_));
}

bool perspective::reproject_to_bearing(const Mat33_t& rot_cw, const Vec3_t& trans_cw, const Vec3_t& pos_w, Vec3_t& reproj) const {
    // convert to camera-coordinates
    reproj = rot_cw * pos_w + trans_cw;

    // check if the point is visible
    if (reproj(2) <= 0.0) {
        return false;
    }

    // reproject onto the image
    const auto z_inv = 1.0 / reproj(2);
    const auto x = fx_ * reproj(0) * z_inv + cx_;
    const auto y = fy_ * reproj(1) * z_inv + cy_;

    // convert to a bearing
    reproj.normalize();

    // check if the point is visible
    return (img_bounds_.min_x_ < x && x < img_bounds_.max_x_
            && img_bounds_.min_y_ < y && y < img_bounds_.max_y_);
}

nlohmann::json perspective::to_json() const {
    return {{"model_type", get_model_type_string()},
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
            {"k1", k1_},
            {"k2", k2_},
            {"p1", p1_},
            {"p2", p2_},
            {"k3", k3_}};
}

void perspective::jacobian_xyz_to_cam(const Vec3_t &xyz,
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
    const double t17 = t16*t16;
    const double t18 = X*2.0;
    const double t19 = tx*2.0;
    const double t20 = Z*ry*2.0;
    const double t22 = Y*rz*2.0;
    const double t21 = t18+t19+t20-t22;
    const double t23 = 1.0/t5;
    const double t24 = Y*2.0;
    const double t25 = ty*2.0;
    const double t26 = X*rz*2.0;
    const double t28 = Z*rx*2.0;
    const double t27 = t24+t25+t26-t28;
    const double t29 = 1.0/(t5*t5*t5);
    const double t30 = t14*t29*2.0;
    const double t31 = t9*t29*2.0;
    const double t32 = t30+t31;
    const double t33 = k1_*t16;
    const double t34 = k2_*t17;
    const double t35 = k3_*t16*t17;
    const double t36 = t33+t34+t35+1.0;
    const double t37 = Z*t3*t6*2.0;
    const double t38 = Y*t14*t29*2.0;
    const double t39 = Y*t9*t29*2.0;
    const double t40 = t37+t38+t39;
    const double t41 = X*t14*t29*2.0;
    const double t42 = Z*t2*t6*2.0;
    const double t43 = X*t9*t29*2.0;
    const double t44 = t41+t42+t43;
    const double t45 = X*t3*t6*2.0;
    const double t47 = Y*t2*t6*2.0;
    const double t46 = t45-t47;
    const double t48 = k1_*t6*t21;
    const double t49 = k2_*t6*t16*t21*2.0;
    const double t50 = k3_*t6*t17*t21*3.0;
    const double t51 = t48+t49+t50;
    const double t52 = t23*t36;
    const double t53 = k1_*t6*t27;
    const double t54 = k2_*t6*t16*t27*2.0;
    const double t55 = k3_*t6*t17*t27*3.0;
    const double t56 = t53+t54+t55;
    const double t57 = k1_*t32;
    const double t58 = k2_*t16*t32*2.0;
    const double t59 = k3_*t17*t32*3.0;
    const double t60 = t57+t58+t59;
    const double t61 = Z*t23*t36;
    const double t62 = k1_*t40;
    const double t63 = k3_*t17*t40*3.0;
    const double t64 = k2_*t16*t40*2.0;
    const double t65 = t62+t63+t64;
    const double t66 = k1_*t44;
    const double t67 = k3_*t17*t44*3.0;
    const double t68 = k2_*t16*t44*2.0;
    const double t69 = t66+t67+t68;
    const double t70 = k1_*t46;
    const double t71 = k2_*t16*t46*2.0;
    const double t72 = k3_*t17*t46*3.0;
    const double t73 = t70+t71+t72;
    const double fx_scaled = scale * fx_;
    const double fy_scaled = scale * fy_;
    jac.setZero();
    jac(0,0) = fx_scaled*(t52+p1_*t3*t6*2.0+p2_*t6*t21*3.0+t2*t23*t51);
    jac(0,1) = fx_scaled*(p1_*t2*t6*2.0+p2_*t6*t27+t2*t23*t56);
    jac(0,2) = -fx_scaled*(p2_*(t30+t9*t29*6.0)+t2*t6*t36+t2*t23*t60+p1_*t2*t3*t29*4.0);
    jac(0,3) = -fx_scaled*(p2_*(t37+t38+Y*t9*t29*6.0)+t2*t23*t65+Z*p1_*t2*t6*2.0+Y*t2*t6*t36+Y*p1_*t2*t3*t29*4.0);
    jac(0,4) = fx_scaled*(t61+p2_*(t41+X*t9*t29*6.0+Z*t2*t6*6.0)+t2*t23*t69+Z*p1_*t3*t6*2.0+X*t2*t6*t36+X*p1_*t2*t3*t29*4.0);
    jac(0,5) = fx_scaled*(p2_*(t45-Y*t2*t6*6.0)-Y*t23*t36+t2*t23*t73+X*p1_*t2*t6*2.0-Y*p1_*t3*t6*2.0);
    jac(1,0) = fy_scaled*(p2_*t3*t6*2.0+p1_*t6*t21+t3*t23*t51);
    jac(1,1) = fy_scaled*(t52+p2_*t2*t6*2.0+p1_*t6*t27*3.0+t3*t23*t56);
    jac(1,2) = -fy_scaled*(p1_*(t31+t14*t29*6.0)+t3*t6*t36+t3*t23*t60+p2_*t2*t3*t29*4.0);
    jac(1,3) = -fy_scaled*(t61+p1_*(t39+Y*t14*t29*6.0+Z*t3*t6*6.0)+t3*t23*t65+Z*p2_*t2*t6*2.0+Y*t3*t6*t36+Y*p2_*t2*t3*t29*4.0);
    jac(1,4) = fy_scaled*(p1_*(t42+t43+X*t14*t29*6.0)+t3*t23*t69+Z*p2_*t3*t6*2.0+X*t3*t6*t36+X*p2_*t2*t3*t29*4.0);
    jac(1,5) = fy_scaled*(-p1_*(t47-X*t3*t6*6.0)+X*t23*t36+t3*t23*t73+X*p2_*t2*t6*2.0-Y*p2_*t3*t6*2.0);
    jac *= -1.0;
//     for (int i=0; i < 1000; ++i) {

//    const cv::Vec3d xyz_cv(xyz[0], xyz[1], xyz[2]);
//    const cv::Vec3d rvec(0.0,0.0,0.0);
//    const cv::Vec3d tvec(0.0,0.0,0.0);
//    cv::Mat im_pts;
//    cv::Mat jacobian;
//    cv::projectPoints(xyz_cv, rvec, tvec, cv_cam_matrix_ * scale, cv_dist_params_, im_pts, jacobian);
//    jac(0,0) = -jacobian.ptr<double>(0)[3];
//    jac(0,1) = -jacobian.ptr<double>(0)[4];
//    jac(0,2) = -jacobian.ptr<double>(0)[5];
//    jac(0,3) = -jacobian.ptr<double>(0)[0];
//    jac(0,4) = -jacobian.ptr<double>(0)[1];
//    jac(0,5) = -jacobian.ptr<double>(0)[2];

//    jac(1,0) = -jacobian.ptr<double>(1)[3];
//    jac(1,1) = -jacobian.ptr<double>(1)[4];
//    jac(1,2) = -jacobian.ptr<double>(1)[5];
//    jac(1,3) = -jacobian.ptr<double>(1)[0];
//    jac(1,4) = -jacobian.ptr<double>(1)[1];
//    jac(1,5) = -jacobian.ptr<double>(1)[2];
//     }
//    t2 = time_now::now();
//    std::cout<<"jacs ocv: "<<jac<<std::endl;

//    std::cout << "eval time ocv jac : "
//        << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
//        << " ms\n";
}

} // namespace camera
} // namespace openvslam
