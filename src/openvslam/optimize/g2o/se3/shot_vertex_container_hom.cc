#include "openvslam/optimize/g2o/se3/shot_vertex_container_hom.h"
#include "openvslam/util/converter.h"
#include "openvslam/optimize/g2o/se3/SE3hom.h"

namespace openvslam {
namespace optimize {
namespace g2o {
namespace se3 {

shot_vertex_hom_container::shot_vertex_hom_container(const unsigned int offset, const unsigned int num_reserve)
    : offset_(offset) {
    vtx_container_.reserve(num_reserve);
}

shot_vertex_hom* shot_vertex_hom_container::create_vertex(const unsigned int id, const Mat44_t& cam_pose_cw, const bool is_constant) {
    // vertexを作成
    const auto vtx_id = offset_ + id;
    auto vtx = new shot_vertex_hom();
    vtx->setId(vtx_id);
    Vec6_t pose_cw;
    world_2_cam_trafo_to_vec_6(cam_pose_cw, pose_cw);
    vtx->setEstimate(pose_cw);
    vtx->setFixed(is_constant);
    // databaseに登録
    vtx_container_[id] = vtx;
    // max IDを更新
    if (max_vtx_id_ < vtx_id) {
        max_vtx_id_ = vtx_id;
    }
    // 作成したvertexをreturn
    return vtx;
}

} // namespace se3
} // namespace g2o
} // namespace optimize
} // namespace openvslam
