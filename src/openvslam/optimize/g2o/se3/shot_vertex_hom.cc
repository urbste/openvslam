#include "openvslam/optimize/g2o/se3/shot_vertex_hom.h"

namespace openvslam {
namespace optimize {
namespace g2o {
namespace se3 {

shot_vertex_hom::shot_vertex_hom()
    : BaseVertex<6, Vec6_t>() {}

bool shot_vertex_hom::read(std::istream& is) {
    Vec6_t estimate;
    for (unsigned int i = 0; i < 7; ++i) {
        is >> estimate(i);
    }
    return true;
}

bool shot_vertex_hom::write(std::ostream& os) const {
    return true;
}

} // namespace se3
} // namespace g2o
} // namespace optimize
} // namespace openvslam
