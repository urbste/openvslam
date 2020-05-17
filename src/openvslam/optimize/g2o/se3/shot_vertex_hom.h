#ifndef OPENVSLAM_OPTIMIZER_G2O_SE3_SHOT_VERTEX_HOM_H
#define OPENVSLAM_OPTIMIZER_G2O_SE3_SHOT_VERTEX_HOM_H

#include "openvslam/type.h"
#include "openvslam/optimize/g2o/se3/SE3hom.h"
#include <g2o/core/base_vertex.h>
#include <g2o/types/slam3d/se3quat.h>

namespace openvslam {
namespace optimize {
namespace g2o {
namespace se3 {

// The Vec6_t holds the rodrigues vector (3 elements) and the translation vector (3 elements)
// so that the projection from world to camera is: R*X+t
class shot_vertex_hom final : public ::g2o::BaseVertex<6, Vec6_t> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    shot_vertex_hom();

    bool read(std::istream& is) override;

    bool write(std::ostream& os) const override;

    void setToOriginImpl() override {
        _estimate.fill(0);
    }

    void oplusImpl(const number_t* update_) override {
        Eigen::Map<const Vec6_t> update(update_);
        // maybe we should make this a multiplication update
        _estimate += update;
//        Mat44_t dT = Mat44_t::Identity();
//        Mat44_t T = Mat44_t::Identity();
//        Mat33_t R;
//        rotation_from_rodr_vec(_estimate.head(3), R);
//        R_and_t_to_hom(R, _estimate.tail(3), T);
//        // update rotation, see Förstner et al., PCV, page 383
//        Mat33_t RdR;
//        //rotation_from_rodr_vec(update.tail(3), RdR);
//        skew_mat(update.head(3), RdR);

//        // I3 + R(dR)
//        R_and_t_to_hom(dT.block<3,3>(0,0) + RdR, update.tail(3), dT);

//        //std::cout<<"RdR: "<<RdR<<std::endl;
//        const Mat44_t updated_T = dT * T;
//        Vec3_t new_R_min;
//        rotation_to_rodr_vec(updated_T.block<3,3>(0,0), new_R_min);
//        _estimate.head(3) = new_R_min;
//        _estimate.tail(3) = updated_T.block<3,1>(0,3);
    }

};

} // namespace se3
} // namespace g2o
} // namespace optimize
} // namespace openvslam

#endif // OPENVSLAM_OPTIMIZER_G2O_SE3_SHOT_VERTEX_HOM_H
