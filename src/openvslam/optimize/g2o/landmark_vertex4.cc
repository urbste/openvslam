#include "openvslam/optimize/g2o/landmark_vertex4.h"
#include <Eigen/Geometry>

namespace openvslam {
namespace optimize {
namespace g2o {

landmark_vertex4::landmark_vertex4()
    : BaseVertex<3, Vec4_t>() {}

bool landmark_vertex4::read(std::istream& is) {
    for (unsigned int i = 0; i < 3; ++i) {
        is >> _estimate(i);
    }
    _estimate[3] = 1.0;
    _estimate.normalize();
    return true;
}

bool landmark_vertex4::write(std::ostream& os) const {
    const Vec3_t pos_w = estimate().hnormalized();
    for (unsigned int i = 0; i < 3; ++i) {
        os << pos_w(i) << " ";
    }
    return os.good();
}

} // namespace g2o
} // namespace optimize
} // namespace openvslam
