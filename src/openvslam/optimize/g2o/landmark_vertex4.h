#ifndef OPENVSLAM_OPTIMIZER_G2O_LANDMARK_VERTEX4_H
#define OPENVSLAM_OPTIMIZER_G2O_LANDMARK_VERTEX4_H

#include "openvslam/type.h"
#include "openvslam/optimize/g2o/nullspace.h"
#include "openvslam/optimize/ceres/autodiff.h"

#include <g2o/core/base_vertex.h>


namespace openvslam {
namespace optimize {
namespace g2o {

class landmark_vertex4 final : public ::g2o::BaseVertex<3, Vec4_t> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    landmark_vertex4();

    bool read(std::istream& is) override;

    bool write(std::ostream& os) const override;

    void setToOriginImpl() override {
        _estimate[0] = 0.0;
        _estimate[1] = 0.0;
        _estimate[2] = 0.0;
        _estimate[3] = 1.0;
    }


    virtual void oplusImpl(const double* update)
    {
        Eigen::Map<const Vec3_t> v(update);
        //_estimate += v;
        // same as
         _estimate += nullSpace*v;
         _estimate.normalize();
//        const double x1 = _estimate[0];
//        const double x2 = _estimate[1];
//        const double x3 = _estimate[2];
//        const double x4 = _estimate[3];

//        if (_estimate[3] > 0.0) {
//            const double denom = 1.0 + x4;
//            const double x1x2_denom = - (x1*x2)/denom;
//            const double x1x3_denom = - (x1*x3)/denom;
//            const double x2x3_denom = - (x2*x3)/denom;

//            _estimate[0] += (  v[0] * (1.0 - (x1*x1) / denom)
//                             + v[1] * x1x2_denom
//                             + v[2] * x1x3_denom);

//            _estimate[1] += (  v[0] * x1x2_denom
//                             + v[1] * (1.0 - (x2*x2) / denom)
//                             + v[2] * x2x3_denom);

//            _estimate[2] += (  v[0] * x1x3_denom
//                             + v[1] * x2x3_denom
//                             + v[2] * (1.0 - (x3*x3) / denom));
//            _estimate[3] += (-x1*v[0] - x2*v[1] - x3*v[2]);

//        } else {
//            const double denom = 1.0 - x4;
//            const double x1x2_denom = - (x1*x2)/denom;
//            const double x1x3_denom = - (x1*x3)/denom;
//            const double x2x3_denom = - (x2*x3)/denom;

//            _estimate[0] += (  v[0] * (1.0 - (x1*x1) / denom)
//                             + v[1] * x1x2_denom
//                             + v[2] * x1x3_denom);

//            _estimate[1] += (  v[0] * x1x2_denom
//                             + v[1] * (1.0 - (x2*x2) / denom)
//                             + v[2] * x2x3_denom);

//            _estimate[2] += (  v[0] * x1x3_denom
//                             + v[1] * x2x3_denom
//                             + v[2] * (1.0 - (x3*x3) / denom));
//            _estimate[3] += (x1*v[0] + x2*v[1] + x3*v[2]);
//        }

//        const double _estimate_length = ceres::sqrt(_estimate[0]*_estimate[0] +
//                _estimate[1]*_estimate[1] +
//                _estimate[2]*_estimate[2] +
//                _estimate[3]*_estimate[3]);

//        _estimate /= _estimate_length;
    }

    void updateNullSpace()
    {
        nullS_3x4_templated<double>(_estimate, nullSpace);
    }

    // this needs to be updated before each iteration
    nullspace43_t nullSpace;
};

} // namespace g2o
} // namespace optimize
} // namespace openvslam

#endif // OPENVSLAM_OPTIMIZER_G2O_LANDMARK_VERTEX4_H
