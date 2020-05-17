#include "openvslam/optimize/g2o/se3/perspective_nullspace_reproj_edge.h"

namespace openvslam {
namespace optimize {
namespace g2o {
namespace se3 {

mono_perspective_nullspace_reproj_edge::mono_perspective_nullspace_reproj_edge()
    : BaseBinaryEdge<2, Vec3_t, shot_vertex_hom, landmark_vertex4>() {}

bool mono_perspective_nullspace_reproj_edge::read(std::istream& is) {
    for (unsigned int i = 0; i < 2; ++i) {
        is >> _measurement(i);
    }
    for (unsigned int i = 0; i < 2; ++i) {
        for (unsigned int j = i; j < 2; ++j) {
            is >> information()(i, j);
            if (i != j) {
                information()(j, i) = information()(i, j);
            }
        }
    }
    return true;
}

bool mono_perspective_nullspace_reproj_edge::write(std::ostream& os) const {
    for (unsigned int i = 0; i < 2; ++i) {
        os << measurement()(i) << " ";
    }
    for (unsigned int i = 0; i < 2; ++i) {
        for (unsigned int j = i; j < 2; ++j) {
            os << " " << information()(i, j);
        }
    }
    return os.good();
}

//void mono_perspective_nullspace_reproj_edge::linearizeOplus() {
//    auto pose = static_cast<shot_vertex_hom*>(_vertices.at(1));
//    auto point = static_cast<landmark_vertex4*>(_vertices.at(0));

//    const Vec6_t min_vec = pose->estimate();
//    const Vec4_t point_est = point->estimate();

//    Eigen::Matrix<double, 3, 4> P;
//    P.fill(0.0);
//    P.block<3,3>(0,0) = Mat33_t::Identity();
//    Mat44_t cam_pose_cw = Mat44_t::Identity();

//    Mat33_t R;
//    rotation_from_rodr_vec(min_vec.head(3), R);
//    cam_pose_cw.block<3,3>(0,0) = R;
//    cam_pose_cw.block<3,1>(0,3) = min_vec.tail(3);

//    // project point to cam and normalize to create new observation
//    // this will create x_i^a in eq. 12.225
//    Vec3_t x_i_a;
//    project_hom_point_to_camera(min_vec, point_est, x_i_a);

//    // now we need the nullspace and the jacobian of that point
//    nullspace32_t ns_x_i_a;
//    Mat33_t jac_x_i_a;
//    nullS_3x2_templated<double>(x_i_a, ns_x_i_a);
//    jacobian_3_vec<double>(x_i_a, jac_x_i_a);

//    Eigen::Matrix<double, 2, 4> ns_js_P = ns_x_i_a.transpose() * jac_x_i_a * P;
//    //dx/dX
//    _jacobianOplusXi = ns_js_P * cam_pose_cw * point->nullSpace;
//    Mat33_t skew_;
//    skew_mat(x_i_a, skew_);
//    Eigen::Matrix<double, 4, 6> tmp;
//    tmp.fill(0.0);
//    tmp.block<3,3>(0,0) = -skew_;
//    tmp.block<3,3>(0,3) = Mat33_t::Identity() * point_est(3);
//    //dx/dRt
//    _jacobianOplusXj = ns_js_P * tmp;
//}

} // namespace se3
} // namespace g2o
} // namespace optimize
} // namespace openvslam
