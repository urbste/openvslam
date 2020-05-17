#include "openvslam/optimize/g2o/landmark_vertex4_container.h"

#include <Eigen/Geometry>

namespace openvslam {
namespace optimize {
namespace g2o {

landmark_vertex4_container::landmark_vertex4_container(const unsigned int offset, const unsigned int num_reserve)
    : offset_(offset) {
    vtx_container_.reserve(num_reserve);
}

landmark_vertex4* landmark_vertex4_container::create_vertex(const unsigned int id, const Vec3_t& pos_w, const bool is_constant) {
    // vertexを作成
    const auto vtx_id = offset_ + id;
    auto vtx = new landmark_vertex4();
    vtx->setId(vtx_id);
    // convert landmark to 4-vector and normalize it to 1
    vtx->setEstimate(pos_w.homogeneous().normalized());
    vtx->setFixed(is_constant);
    vtx->setMarginalized(true);
    vtx->updateNullSpace();
    // databaseに登録
    vtx_container_[id] = vtx;
    // max IDを更新
    if (max_vtx_id_ < vtx_id) {
        max_vtx_id_ = vtx_id;
    }
    // 作成したvertexをreturn
    return vtx;
}

} // namespace g2o
} // namespace optimize
} // namespace openvslam
