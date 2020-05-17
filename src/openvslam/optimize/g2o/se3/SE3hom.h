#ifndef OPENVSLAM_OPTIMIZER_G2O_SE3_SE3HOM_H
#define OPENVSLAM_OPTIMIZER_G2O_SE3_SE3HOM_H

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "openvslam/type.h"

namespace openvslam {
namespace optimize {
namespace g2o {
namespace se3 {

void skew_mat(const Vec3_t& v,
          Mat33_t& m);

void deltaR(const Mat33_t& R,
               Vec3_t& v);

void rotation_from_rodr_vec(const Vec3_t& rodrigues_vec,
                           Mat33_t& rotation);

void rotation_to_rodr_vec(const Mat33_t& rotation,
                         Vec3_t& rodrigues_vec);

void world_2_cam_trafo_to_vec_6(const Mat33_t& rot_cw_,
                                const Vec3_t& cam_center_,
                                Vec6_t& vec);

void world_2_cam_trafo_to_vec_6(const Mat44_t trafo_cw_,
                                Vec6_t& vec);

void vec_6_to_world_to_cam(const Vec6_t vec_6,
                                Mat44_t& trafo_cw);

void R_and_t_to_hom(const Mat33_t& R,
                    const Vec3_t& t,
                    Mat44_t &T);

void project_hom_point_to_camera(const Vec6_t& min_trafo,
                          const Vec4_t& homogeneous_pt4,
                          Vec3_t& projected_point);

}
}
}
}

#endif // OPENVSLAM_OPTIMIZER_G2O_SE3_SE3HOM_H
