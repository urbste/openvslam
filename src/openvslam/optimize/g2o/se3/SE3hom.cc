#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iostream>

#include "openvslam/type.h"
#include "openvslam/optimize/g2o/se3/SE3hom.h"

namespace openvslam {
namespace optimize {
namespace g2o {
namespace se3 {

void skew_mat(const Vec3_t& v, Mat33_t& m)
{
   m.fill(0.0);
   m(0,1) = -v(2);
   m(0,2) =  v(1);
   m(1,2) = -v(0);
   m(1,0) =  v(2);
   m(2,0) = -v(1);
   m(2,1) =  v(0);
}

void deltaR(const Mat33_t& R, Vec3_t& v)
{
  v(0) = R(2,1) - R(1,2);
  v(1) = R(0,2) - R(2,0);
  v(2) = R(1,0) - R(0,1);
}

void rotation_from_rodr_vec(const Vec3_t& rodrigues_vec,
                           Mat33_t& rotation)
{
    const double theta = rodrigues_vec.norm();
    Mat33_t Omega;
    skew_mat(rodrigues_vec, Omega);
    const Mat33_t Omega2 = Omega * Omega;

    if (theta < 0.00001)
    {
        rotation = (Mat33_t::Identity()
            + Omega
            + 0.5 * Omega2);
    }
    else {
        rotation = (Mat33_t::Identity()
            + std::sin(theta)/theta *Omega
            + (1.0-std::cos(theta))/(theta*theta)*Omega2);
    }
}

void rotation_to_rodr_vec(const Mat33_t& rotation,
                         Vec3_t& rodrigues_vec) {
    const double d = 0.5*(rotation(0,0)+rotation(1,1)+rotation(2,2)-1.0);


    Vec3_t dR;
    deltaR(rotation, dR);

    if (std::abs(d) > 0.99999)
    {
      rodrigues_vec = 0.5 * dR;
    }
    else
    {
      const double theta = std::acos(d);
      rodrigues_vec = theta/(2.0 * std::sqrt(1.0-d*d))*dR;

    }
}

void world_2_cam_trafo_to_vec_6(const Mat33_t& rot_cw_,
                                const Vec3_t& trans_cw_,
                                Vec6_t& vec) {
    Vec3_t rodrigues;
    rotation_to_rodr_vec(rot_cw_, rodrigues);
    vec.head(3) = rodrigues;
    vec.tail(3) = trans_cw_;
}

void world_2_cam_trafo_to_vec_6(const Mat44_t trafo_cw_,
                                Vec6_t& vec) {
    Vec3_t rodrigues;
    rotation_to_rodr_vec(trafo_cw_.block<3,3>(0,0),rodrigues);
    vec.head(3) = rodrigues;
    vec.tail(3) = trafo_cw_.block<3,1>(0,3);
}

void vec_6_to_world_to_cam(const Vec6_t vec_6,
                           Mat44_t& trafo_cw) {

    trafo_cw = Mat44_t::Identity();
    Mat33_t rotation;
    rotation_from_rodr_vec(vec_6.head(3), rotation);
    trafo_cw.block<3, 3>(0, 0) = rotation;
    trafo_cw.block<3, 1>(0, 3) = vec_6.tail(3);
}

void R_and_t_to_hom(const Mat33_t& R,
                    const Vec3_t& t,
                    Mat44_t& T) {
    T = Mat44_t::Identity();
    T.block<3,3>(0,0) = R;
    T.block<3,1>(0,3) = t;
}

void project_hom_point_to_camera(const Vec6_t& min_trafo,
                          const Vec4_t& homogeneous_pt4,
                          Vec3_t& projected_point) {
    Mat33_t rotation;
    rotation_from_rodr_vec(min_trafo.head(3), rotation);
    Mat44_t T;
    R_and_t_to_hom(rotation, min_trafo.tail(3), T);
    projected_point = (T * homogeneous_pt4).hnormalized().normalized();
    //projected_point = (rotation.transpose() *
    //        (homogeneous_pt4.head(3) - homogeneous_pt4(3)*min_trafo.tail(3))).normalized();
}




}
}
}
}


