#include "openvslam/data/keyframe.h"
#include "openvslam/data/landmark.h"
#include "openvslam/data/map_database.h"
#include "openvslam/optimize/global_bundle_adjuster.h"
#ifdef USE_HOMOGENEOUS_LANDMARKS
#include "openvslam/optimize/g2o/nullspace_updater.h"
#include "openvslam/optimize/g2o/landmark_vertex4_container.h"
#include "openvslam/optimize/g2o/landmark_vertex4.h"
#include "openvslam/optimize/g2o/se3/shot_vertex_container_hom.h"
#else
#include "openvslam/optimize/g2o/landmark_vertex_container.h"
#include "openvslam/optimize/g2o/se3/shot_vertex_container.h"
#endif
#include "openvslam/optimize/g2o/se3/reproj_edge_wrapper.h"
#include "openvslam/util/converter.h"

#include <g2o/core/solver.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/sparse_optimizer_terminate_action.h>

namespace openvslam {
namespace optimize {

//void evaluateJacobian(g2o::se3::mono_perspective_nullspace_reproj_edge* e,
//                           ::g2o::JacobianWorkspace& jacobianWorkspace,
//                           ::g2o::JacobianWorkspace& numericJacobianWorkspace)
//{
//  // calling the analytic Jacobian but writing to the numeric workspace
//  e->linearizeOplus(numericJacobianWorkspace);
////  // copy result into analytic workspace
////  jacobianWorkspace = numericJacobianWorkspace;

////  // compute the numeric Jacobian into the numericJacobianWorkspace workspace as setup by the previous call
//  Eigen::Matrix<double, 2, 9> jac;
//  e->test_jac(jac);

//  // compare the Jacobians
//  double* pt = numericJacobianWorkspace.workspaceForVertex(0);
//  double* pose = numericJacobianWorkspace.workspaceForVertex(1);
//  //numElems *= EdgeType1::VertexXiType::Dimension;
//  Eigen::Map<Eigen::Matrix<double, 2, 6>> jac_num_pose(pose);
//  Eigen::Map<Eigen::Matrix<double, 2, 3>> jac_num_pt(pt);
//  std::cout<<"jac_num pose: \n"<<jac_num_pose<<std::endl;
//  std::cout<<"jac_num point: \n"<<jac_num_pt<<std::endl;
//  std::cout<<"jac analytic: \n"<<jac<<std::endl;
//  //int numElems = EdgeType1::Dimension;


//}


global_bundle_adjuster::global_bundle_adjuster(data::map_database* map_db, const unsigned int num_iter, const bool use_huber_kernel)
    : map_db_(map_db), num_iter_(num_iter), use_huber_kernel_(use_huber_kernel) {}

void global_bundle_adjuster::optimize(const unsigned int lead_keyfrm_id_in_global_BA, bool* const force_stop_flag) const {
    // 1. データを集める

    const auto keyfrms = map_db_->get_all_keyframes();
    const auto lms = map_db_->get_all_landmarks();
    std::vector<bool> is_optimized_lm(lms.size(), true);

    // 2. optimizerを構築
    auto linear_solver = ::g2o::make_unique<::g2o::LinearSolverCSparse<::g2o::BlockSolver_6_3::PoseMatrixType>>();
    auto block_solver = ::g2o::make_unique<::g2o::BlockSolver_6_3>(std::move(linear_solver));
    auto algorithm = new ::g2o::OptimizationAlgorithmLevenberg(std::move(block_solver));

    ::g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(algorithm);
    optimizer.setVerbose(true);
#ifdef USE_HOMOGENEOUS_LANDMARKS
    //algorithm->setUserLambdaInit(1e-4);
#endif
    if (force_stop_flag) {
        optimizer.setForceStopFlag(force_stop_flag);
    }

    // 3. keyframeをg2oのvertexに変換してoptimizerにセットする

    // shot vertexのcontainer
#ifdef USE_HOMOGENEOUS_LANDMARKS
    g2o::se3::shot_vertex_hom_container keyfrm_vtx_container(0, keyfrms.size());
    size_t cont_vtx_id = 0;
#else
    g2o::se3::shot_vertex_container keyfrm_vtx_container(0, keyfrms.size());
#endif
    // keyframesをoptimizerにセット
    for (const auto keyfrm : keyfrms) {
        if (!keyfrm) {
            continue;
        }
        if (keyfrm->will_be_erased()) {
            continue;
        }

        auto keyfrm_vtx = keyfrm_vtx_container.create_vertex(keyfrm, keyfrm->id_ == 0);

        optimizer.addVertex(keyfrm_vtx);
#ifdef USE_HOMOGENEOUS_LANDMARKS
        ++cont_vtx_id;
#endif
    }

    // 4. keyframeとlandmarkのvertexをreprojection edgeで接続する

    // landmark vertexのcontainer
#ifdef USE_HOMOGENEOUS_LANDMARKS
    g2o::landmark_vertex4_container lm_vtx_container(keyfrm_vtx_container.get_max_vertex_id() + 1, lms.size());
#else
    g2o::landmark_vertex_container lm_vtx_container(keyfrm_vtx_container.get_max_vertex_id() + 1, lms.size());
#endif
    // reprojection edgeのcontainer
    using reproj_edge_wrapper = g2o::se3::reproj_edge_wrapper<data::keyframe>;
    std::vector<reproj_edge_wrapper> reproj_edge_wraps;
    reproj_edge_wraps.reserve(10 * lms.size());

    constexpr float chi_sq_2D = 5.99146;
    constexpr float chi_sq_3D = 7.81473;
    const float sqrt_chi_sq_2D = std::sqrt(chi_sq_2D);
    const float sqrt_chi_sq_3D = std::sqrt(chi_sq_3D);

#ifdef USE_HOMOGENEOUS_LANDMARKS
    std::unordered_map<unsigned int, data::landmark*> local_lms;
    std::vector<size_t> lm_ids_in_optimizer;
#endif
    std::map<unsigned int, Vec3_t> lm_id_to_coord;
    for (unsigned int i = 0; i < lms.size(); ++i) {
        auto lm = lms.at(i);
        if (!lm) {
            continue;
        }
        if (lm->will_be_erased()) {
            continue;
        }
#ifdef USE_HOMOGENEOUS_LANDMARKS

        local_lms[lm->id_] = lm;
#endif
        // landmarkをg2oのvertexに変換してoptimizerにセットする
        auto lm_vtx = lm_vtx_container.create_vertex(lm, false);
        lm_id_to_coord.insert(std::pair<unsigned int, Vec3_t>(lm->get_index_in_keyframe(keyfrms[0]), lm->get_pos_in_world()));
        optimizer.addVertex(lm_vtx);
#ifdef USE_HOMOGENEOUS_LANDMARKS
        lm_ids_in_optimizer.push_back(cont_vtx_id);
        ++cont_vtx_id;
#endif

        unsigned int num_edges = 0;
        const auto observations = lm->get_observations();
        for (const auto& obs : observations) {
            auto keyfrm = obs.first;
            auto idx = obs.second;
            if (!keyfrm) {
                continue;
            }
            if (keyfrm->will_be_erased()) {
                continue;
            }

            if (!keyfrm_vtx_container.contain(keyfrm)) {
                continue;
            }

            const auto keyfrm_vtx = keyfrm_vtx_container.get_vertex(keyfrm);

            const auto& undist_keypt = keyfrm->undist_keypts_.at(idx);
            const auto& bearing = keyfrm->bearings_.at(idx);
            const auto& bearing_jac = keyfrm->bearings_jac_.at(idx);
            const auto& bearing_ns = keyfrm->bearings_nullspace_.at(idx);
            const float x_right = keyfrm->stereo_x_right_.at(idx);
            const float inv_sigma_sq = keyfrm->inv_level_sigma_sq_.at(undist_keypt.octave);
            const auto sqrt_chi_sq = (keyfrm->camera_->setup_type_ == camera::setup_type_t::Monocular)
                                         ? sqrt_chi_sq_2D
                                         : sqrt_chi_sq_3D;
#ifdef USE_HOMOGENEOUS_LANDMARKS
            auto reproj_edge_wrap = reproj_edge_wrapper(keyfrm, keyfrm_vtx, lm, lm_vtx,
                                                        bearing, bearing_jac, bearing_ns,
                                                        idx, undist_keypt.pt.x, undist_keypt.pt.y, x_right,
                                                        inv_sigma_sq, sqrt_chi_sq, use_huber_kernel_);
#else
            auto reproj_edge_wrap = reproj_edge_wrapper(keyfrm, keyfrm_vtx, lm, lm_vtx,
                                                        idx, undist_keypt.pt.x, undist_keypt.pt.y, x_right,
                                                        inv_sigma_sq, sqrt_chi_sq, use_huber_kernel_);
#endif
            reproj_edge_wraps.push_back(reproj_edge_wrap);
            optimizer.addEdge(reproj_edge_wrap.edge_);
            ++num_edges;
//            if (!keyfrm_vtx->fixed()) {
//            numericJacobianWorkspace.updateSize(reproj_edge_wrap.edge_);
//            numericJacobianWorkspace.allocate();
//            evaluateJacobian(static_cast<g2o::se3::mono_perspective_nullspace_reproj_edge*>(
//                                      reproj_edge_wrap.edge_),
//                                  jacobianWorkspace, numericJacobianWorkspace);
//            }
        }

        if (num_edges == 0) {
            optimizer.removeVertex(lm_vtx);
            is_optimized_lm.at(i) = false;
        }
    }


    ::g2o::SparseOptimizerTerminateAction* terminateAction = 0;
        terminateAction = new ::g2o::SparseOptimizerTerminateAction;
    terminateAction->setGainThreshold(1e-6);
    optimizer.addPostIterationAction(terminateAction);

    optimizer.initializeOptimization();
#ifdef USE_HOMOGENEOUS_LANDMARKS
    double chi2 = std::numeric_limits<double>::max();
    for (int i=0; i < num_iter_; ++i) {
        optimizer.optimize(1);
        const double new_chi2 = optimizer.activeRobustChi2();
        if (std::abs(chi2-new_chi2) < 1e-6) {
            break;
        } else {
            chi2 = new_chi2;
        }
        // update nullspaces
        for (auto id_local_lm_pair : local_lms) {
            auto local_lm = id_local_lm_pair.second;
            g2o::landmark_vertex4* lm = lm_vtx_container.get_vertex(local_lm);
            if (lm)
                if (!lm->fixed())
                    lm->updateNullSpace();
        }
    }
#else
    optimizer.optimize(num_iter_);
#endif

    if (force_stop_flag && *force_stop_flag) {
        return;
    }

    // 6. 結果を取り出す

    for (auto keyfrm : keyfrms) {
        if (keyfrm->will_be_erased()) {
            continue;
        }
        auto keyfrm_vtx = keyfrm_vtx_container.get_vertex(keyfrm);
#ifdef   USE_HOMOGENEOUS_LANDMARKS
        Mat44_t cam_pose_cw;
        g2o::se3::vec_6_to_world_to_cam(keyfrm_vtx->estimate(), cam_pose_cw);
#else
        const auto cam_pose_cw = util::converter::to_eigen_mat(keyfrm_vtx->estimate());
#endif
        if (lead_keyfrm_id_in_global_BA == 0) {
            keyfrm->set_cam_pose(cam_pose_cw);
        }
        else {
            keyfrm->cam_pose_cw_after_loop_BA_ = cam_pose_cw;
            keyfrm->loop_BA_identifier_ = lead_keyfrm_id_in_global_BA;
        }
    }

    for (unsigned int i = 0; i < lms.size(); ++i) {
        if (!is_optimized_lm.at(i)) {
            continue;
        }

        auto lm = lms.at(i);
        if (!lm) {
            continue;
        }
        if (lm->will_be_erased()) {
            continue;
        }
        auto lm_vtx = lm_vtx_container.get_vertex(lm);
#ifdef USE_HOMOGENEOUS_LANDMARKS
        const Vec3_t pos_w = lm_vtx->estimate().hnormalized();
#else
        const Vec3_t pos_w = lm_vtx->estimate();
#endif
        //const Vec3_t diff = lm_id_to_coord.find(lm->get_index_in_keyframe(keyfrms[0]))->second - pos_w;

        if (lead_keyfrm_id_in_global_BA == 0) {
            lm->set_pos_in_world(pos_w);
            lm->update_normal_and_depth();
        }
        else {
            lm->pos_w_after_global_BA_ = pos_w;
            lm->loop_BA_identifier_ = lead_keyfrm_id_in_global_BA;
        }
    }
}

} // namespace optimize
} // namespace openvslam
