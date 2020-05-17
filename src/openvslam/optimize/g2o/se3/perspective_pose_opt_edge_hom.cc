#include "openvslam/optimize/g2o/se3/perspective_pose_opt_edge_hom.h"
#include "openvslam/optimize/g2o/nullspace.h"
#include "openvslam/optimize/g2o/se3/SE3hom.h"

namespace openvslam {
namespace optimize {
namespace g2o {
namespace se3 {

mono_perspective_pose_opt_edge_hom::mono_perspective_pose_opt_edge_hom()
    : ::g2o::BaseUnaryEdge<2, Vec3_t, shot_vertex_hom>() {}

bool mono_perspective_pose_opt_edge_hom::read(std::istream& is) {
    for (unsigned int i = 0; i < 3; ++i) {
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

bool mono_perspective_pose_opt_edge_hom::write(std::ostream& os) const {
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


// see Förstner et al, PCV page 520
//void mono_perspective_pose_opt_edge_hom::linearizeOplus() {
////    auto vi = static_cast<shot_vertex_hom*>(_vertices.at(0));
////    Mat33_t R;
////    const Vec6_t min_vec = vi->shot_vertex_hom::estimate();
////    rotation_from_rodr_vec(min_vec.tail(3), R);

////    // project point to cam and normalize to create new observation
////    // this will create x_i^a in eq. 12.225
////    Vec3_t x_i_a;
////    project_hom_point_to_camera(min_vec, pos_w_hom_, x_i_a);

////    // now we need the nullspace and the jacobian of that point
////    nullspace32_t ns_x_i_a;
////    Mat33_t jac_x_i_a;
////    nullS_3x2_templated<double>(x_i_a, ns_x_i_a);
////    jacobian_3_vec<double>(x_i_a, jac_x_i_a);

////    Eigen::Matrix<double, 2, 3> ns_js = ns_x_i_a.transpose() * jac_x_i_a;

////    _jacobianOplusXi.block<2,3>(0,0) = ns_js * R * pos_w_hom_(3);
////    Vec3_t tmp1 = R * (pos_w_hom_.head(3) - pos_w_hom_(3) * min_vec.head(3));
////    Mat33_t skew_;
////    skew_mat(tmp1, skew_);
////    _jacobianOplusXi.block<2,3>(0,3) = ns_js*skew_;
//    auto vi = static_cast<shot_vertex_hom*>(_vertices.at(0));
//    Eigen::Matrix<double, 3, 4> P;
//    P.fill(0.0);
//    P.block<3,3>(0,0) = Mat33_t::Identity();
//    //Mat44_t cam_pose_cw = Mat44_t::Identity();

//    //Mat33_t R;
//    const Vec6_t min_vec = vi->shot_vertex_hom::estimate();

//    // project point to cam and normalize to create new observation
//    // this will create x_i^a in eq. 12.225
//    Vec3_t x_i_a;
//    project_hom_point_to_camera(min_vec, pos_w_hom_, x_i_a);

//    // now we need the nullspace and the jacobian of that point
//    nullspace32_t ns_x_i_a;
//    Mat33_t jac_x_i_a;
//    nullS_3x2_templated<double>(x_i_a, ns_x_i_a);
//    jacobian_3_vec<double>(x_i_a, jac_x_i_a);

//    Eigen::Matrix<double, 2, 4> ns_js_P = ns_x_i_a.transpose() * jac_x_i_a * P;

//    Mat33_t skew_;
//    skew_mat(x_i_a, skew_);
//    Eigen::Matrix<double, 4, 6> tmp;
//    tmp.fill(0.0);
//    tmp.block<3,3>(0,0) = -skew_;
//    tmp.block<3,3>(0,3) = Mat33_t::Identity() * pos_w_hom_(3);
//    _jacobianOplusXi = ns_js_P * tmp;
//}


} // namespace se3
} // namespace g2o
} // namespace optimize
} // namespace openvslam
