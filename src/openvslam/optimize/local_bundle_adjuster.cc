#include "openvslam/data/keyframe.h"
#include "openvslam/data/landmark.h"
#include "openvslam/data/map_database.h"
#include "openvslam/optimize/local_bundle_adjuster.h"
#include "openvslam/util/timer.h"
#ifdef USE_HOMOGENEOUS_LANDMARKS
#include "openvslam/optimize/g2o/nullspace_updater.h"
#include "openvslam/optimize/g2o/landmark_vertex4_container.h"
#include "openvslam/optimize/g2o/se3/shot_vertex_container_hom.h"
#else
#include "openvslam/optimize/g2o/landmark_vertex_container.h"
#include "openvslam/optimize/g2o/se3/shot_vertex_container.h"
#endif
#include "openvslam/optimize/g2o/se3/reproj_edge_wrapper.h"
#include "openvslam/util/converter.h"

#include <unordered_map>

#include <Eigen/StdVector>
#include <g2o/core/solver.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/sparse_optimizer_terminate_action.h>

namespace openvslam {
namespace optimize {

local_bundle_adjuster::local_bundle_adjuster(const unsigned int num_first_iter,
                                             const unsigned int num_second_iter)
    : num_first_iter_(num_first_iter), num_second_iter_(num_second_iter) {}

void local_bundle_adjuster::optimize(openvslam::data::keyframe* curr_keyfrm, bool* const force_stop_flag) const {
    // 1. local/fixed keyframes, local landmarksを集計する

    util::Timer entire_time;
    // correct local keyframes of the current keyframe
    std::unordered_map<unsigned int, data::keyframe*> local_keyfrms;

    local_keyfrms[curr_keyfrm->id_] = curr_keyfrm;
    const auto curr_covisibilities = curr_keyfrm->graph_node_->get_covisibilities();
    for (auto local_keyfrm : curr_covisibilities) {
        if (!local_keyfrm) {
            continue;
        }
        if (local_keyfrm->will_be_erased()) {
            continue;
        }

        local_keyfrms[local_keyfrm->id_] = local_keyfrm;
    }

    // correct local landmarks seen in local keyframes
    std::unordered_map<unsigned int, data::landmark*> local_lms;

    for (auto local_keyfrm : local_keyfrms) {
        const auto landmarks = local_keyfrm.second->get_landmarks();
        for (auto local_lm : landmarks) {
            if (!local_lm) {
                continue;
            }
            if (local_lm->will_be_erased()) {
                continue;
            }

            // 重複を避ける
            if (local_lms.count(local_lm->id_)) {
                continue;
            }

            local_lms[local_lm->id_] = local_lm;
        }
    }

    // fixed keyframes: keyframes which observe local landmarks but which are NOT in local keyframes
    std::unordered_map<unsigned int, data::keyframe*> fixed_keyfrms;
    for (auto local_lm : local_lms) {
        const auto observations = local_lm.second->get_observations();
        for (auto& obs : observations) {
            auto fixed_keyfrm = obs.first;
            if (!fixed_keyfrm) {
                continue;
            }
            if (fixed_keyfrm->will_be_erased()) {
                continue;
            }

            // local keyframesに属しているときは追加しない
            if (local_keyfrms.count(fixed_keyfrm->id_)) {
                continue;
            }

            // 重複を避ける
            if (fixed_keyfrms.count(fixed_keyfrm->id_)) {
                continue;
            }

            fixed_keyfrms[fixed_keyfrm->id_] = fixed_keyfrm;
        }
    }

    // 2. optimizerを構築
    auto linear_solver = ::g2o::make_unique<::g2o::LinearSolverCSparse<::g2o::BlockSolver_6_3::PoseMatrixType>>();
    auto block_solver = ::g2o::make_unique<::g2o::BlockSolver_6_3>(std::move(linear_solver));
    auto algorithm = new ::g2o::OptimizationAlgorithmLevenberg(std::move(block_solver));

    ::g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(algorithm);
    //optimizer.setVerbose(true);
#ifdef USE_HOMOGENEOUS_LANDMARKS
    //algorithm->setUserLambdaInit(1e-4);
#endif
    if (force_stop_flag) {
        optimizer.setForceStopFlag(force_stop_flag);
    }

    // 3. keyframeをg2oのvertexに変換してoptimizerにセットする

    // shot vertexのcontainer
#ifdef USE_HOMOGENEOUS_LANDMARKS
    g2o::se3::shot_vertex_hom_container keyfrm_vtx_container(0, local_keyfrms.size() + fixed_keyfrms.size());
#else
    g2o::se3::shot_vertex_container keyfrm_vtx_container(0, local_keyfrms.size() + fixed_keyfrms.size());
#endif
// vertexに変換されたkeyframesを保存しておく
    std::unordered_map<unsigned int, data::keyframe*> all_keyfrms;
#ifdef USE_HOMOGENEOUS_LANDMARKS
    size_t cont_vtx_id = 0;
#endif

    // local keyframesをoptimizerにセット
    for (auto& id_local_keyfrm_pair : local_keyfrms) {
        auto local_keyfrm = id_local_keyfrm_pair.second;

        all_keyfrms.emplace(id_local_keyfrm_pair);
        auto keyfrm_vtx = keyfrm_vtx_container.create_vertex(local_keyfrm, local_keyfrm->id_ == 0);
        optimizer.addVertex(keyfrm_vtx);
#ifdef USE_HOMOGENEOUS_LANDMARKS
        ++cont_vtx_id;
#endif
    }

    // fixed keyframesをoptimizerにセット
    for (auto& id_fixed_keyfrm_pair : fixed_keyfrms) {
        auto fixed_keyfrm = id_fixed_keyfrm_pair.second;

        all_keyfrms.emplace(id_fixed_keyfrm_pair);
        auto keyfrm_vtx = keyfrm_vtx_container.create_vertex(fixed_keyfrm, true);
        optimizer.addVertex(keyfrm_vtx);
#ifdef USE_HOMOGENEOUS_LANDMARKS
        ++cont_vtx_id;
#endif
    }

    // 4. keyframeとlandmarkのvertexをreprojection edgeで接続する

    // landmark vertexのcontainer
#ifdef USE_HOMOGENEOUS_LANDMARKS
    g2o::landmark_vertex4_container lm_vtx_container(keyfrm_vtx_container.get_max_vertex_id() + 1, local_lms.size());
#else
    g2o::landmark_vertex_container lm_vtx_container(keyfrm_vtx_container.get_max_vertex_id() + 1, local_lms.size());
#endif
    // reprojection edgeのcontainer
    using reproj_edge_wrapper = g2o::se3::reproj_edge_wrapper<data::keyframe>;
    std::vector<reproj_edge_wrapper> reproj_edge_wraps;
    reproj_edge_wraps.reserve(all_keyfrms.size() * local_lms.size());


    constexpr float chi_sq_2D = 5.99146;
    constexpr float chi_sq_3D = 7.81473;

    const float sqrt_chi_sq_2D = std::sqrt(chi_sq_2D);
    const float sqrt_chi_sq_3D = std::sqrt(chi_sq_3D);
#ifdef USE_HOMOGENEOUS_LANDMARKS
    std::vector<size_t> lm_ids_in_optimizer;
#endif
    for (auto& id_local_lm_pair : local_lms) {
        auto local_lm = id_local_lm_pair.second;

        // landmarkをg2oのvertexに変換してoptimizerにセットする
        auto lm_vtx = lm_vtx_container.create_vertex(local_lm, false);
        optimizer.addVertex(lm_vtx);
#ifdef USE_HOMOGENEOUS_LANDMARKS
        lm_ids_in_optimizer.push_back(cont_vtx_id);
        ++cont_vtx_id;
#endif
        const auto observations = local_lm->get_observations();
        for (const auto& obs : observations) {
            auto keyfrm = obs.first;
            auto idx = obs.second;
            if (!keyfrm) {
                continue;
            }
            if (keyfrm->will_be_erased()) {
                continue;
            }

            const auto keyfrm_vtx = keyfrm_vtx_container.get_vertex(keyfrm);
            const auto& undist_keypt = keyfrm->undist_keypts_.at(idx);
            const float x_right = keyfrm->stereo_x_right_.at(idx);
            const float inv_sigma_sq = keyfrm->inv_level_sigma_sq_.at(undist_keypt.octave);
            const auto sqrt_chi_sq = (keyfrm->camera_->setup_type_ == camera::setup_type_t::Monocular)
                                         ? sqrt_chi_sq_2D
                                         : sqrt_chi_sq_3D;
#ifdef USE_HOMOGENEOUS_LANDMARKS
            const auto& bearing = keyfrm->bearings_.at(idx);
            const auto& bearing_jac = keyfrm->bearings_jac_.at(idx);
            const auto& bearing_ns = keyfrm->bearings_nullspace_.at(idx);
            auto reproj_edge_wrap = reproj_edge_wrapper(keyfrm, keyfrm_vtx, local_lm, lm_vtx,
                                                        bearing, bearing_jac, bearing_ns,
                                                        idx, undist_keypt.pt.x, undist_keypt.pt.y, x_right,
                                                        inv_sigma_sq, sqrt_chi_sq);
#else
            auto reproj_edge_wrap = reproj_edge_wrapper(keyfrm, keyfrm_vtx, local_lm, lm_vtx,
                                                        idx, undist_keypt.pt.x, undist_keypt.pt.y, x_right,
                                                        inv_sigma_sq, sqrt_chi_sq);
#endif
            reproj_edge_wraps.push_back(reproj_edge_wrap);
            optimizer.addEdge(reproj_edge_wrap.edge_);
        }
    }

    // 5. 1回目の最適化を実行

    if (force_stop_flag) {
        if (*force_stop_flag) {
            return;
        }
    }

    ::g2o::SparseOptimizerTerminateAction* terminateAction = 0;
        terminateAction = new ::g2o::SparseOptimizerTerminateAction;
    terminateAction->setGainThreshold(1e-4);
    optimizer.addPostIterationAction(terminateAction);
    optimizer.initializeOptimization();
    std::cerr<<"RUNNING LOCAL BUNDLE ADJUSTMENT\n";
#ifdef USE_HOMOGENEOUS_LANDMARKS
    double chi2 = std::numeric_limits<double>::max();
    int nr_iters = 0;
    double chi_diff = 0.0;
    for (unsigned int i = 0; i < num_first_iter_; ++i) {
        optimizer.optimize(1);
        ++nr_iters;
        const double new_chi2 = optimizer.activeRobustChi2();
        chi_diff = std::abs(chi2-new_chi2);
        if (chi_diff < 1e-4) {
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
    std::cout<<"FINISHED FIRST RUN LOCAL BA IN: "<<std::to_string(nr_iters)<<" ITERS\n";
    std::cout<<"LAST DIFF: "<<chi_diff<<std::endl;
#else
    optimizer.optimize(num_first_iter_);
#endif

    // 6. アウトライア除去をして2回目の最適化を実行
    bool run_robust_BA = true;

    if (force_stop_flag) {
        if (*force_stop_flag) {
            run_robust_BA = false;
        }
    }

    if (run_robust_BA) {
        for (auto& reproj_edge_wrap : reproj_edge_wraps) {
            auto edge = reproj_edge_wrap.edge_;

            auto local_lm = reproj_edge_wrap.lm_;
            if (local_lm->will_be_erased()) {
                continue;
            }

            if (reproj_edge_wrap.is_monocular_) {
                if (chi_sq_2D < edge->chi2() || !reproj_edge_wrap.depth_is_positive()) {
                    reproj_edge_wrap.set_as_outlier();
                }
            }
            else {
                if (chi_sq_3D < edge->chi2() || !reproj_edge_wrap.depth_is_positive()) {
                    reproj_edge_wrap.set_as_outlier();
                }
            }

            edge->setRobustKernel(nullptr);
        }

        optimizer.initializeOptimization();
        std::cout<<"RUNNING LOCAL BUNDLE ADJUSTMENT\n";
#ifdef USE_HOMOGENEOUS_LANDMARKS
        chi2 = std::numeric_limits<double>::max();
        nr_iters = 0;
        for (unsigned int  i=0; i < num_second_iter_; ++i) {
            optimizer.optimize(1);
            ++nr_iters;
            const double new_chi2 = optimizer.activeRobustChi2();
            chi_diff = std::abs(chi2-new_chi2);
            if (chi_diff < 1e-4) {
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
        std::cout<<"FINISHED SECOND RUN LOCAL BA IN: "<<std::to_string(nr_iters)<<" ITERS\n";
        std::cout<<"LAST DIFF: "<<chi_diff<<std::endl;

#else
        optimizer.optimize(num_second_iter_);
#endif
    }

    // 7. アウトライアを集計する

    std::vector<std::pair<data::keyframe*, data::landmark*>> outlier_observations;
    outlier_observations.reserve(reproj_edge_wraps.size());

    for (auto& reproj_edge_wrap : reproj_edge_wraps) {
        auto edge = reproj_edge_wrap.edge_;

        auto local_lm = reproj_edge_wrap.lm_;
        if (local_lm->will_be_erased()) {
            continue;
        }

        if (reproj_edge_wrap.is_monocular_) {
            if (chi_sq_2D < edge->chi2() || !reproj_edge_wrap.depth_is_positive()) {
                outlier_observations.emplace_back(std::make_pair(reproj_edge_wrap.shot_, reproj_edge_wrap.lm_));
            }
        }
        else {
            if (chi_sq_3D < edge->chi2() || !reproj_edge_wrap.depth_is_positive()) {
                outlier_observations.emplace_back(std::make_pair(reproj_edge_wrap.shot_, reproj_edge_wrap.lm_));
            }
        }
    }

    // 8. 情報を更新

    {
        std::lock_guard<std::mutex> lock(data::map_database::mtx_database_);

        if (!outlier_observations.empty()) {
            for (auto& outlier_obs : outlier_observations) {
                auto keyfrm = outlier_obs.first;
                auto lm = outlier_obs.second;
                keyfrm->erase_landmark(lm);
                lm->erase_observation(keyfrm);
            }
        }

        for (auto id_local_keyfrm_pair : local_keyfrms) {
            auto local_keyfrm = id_local_keyfrm_pair.second;

            auto keyfrm_vtx = keyfrm_vtx_container.get_vertex(local_keyfrm);
#ifdef USE_HOMOGENEOUS_LANDMARKS
            local_keyfrm->set_cam_pose_min(keyfrm_vtx->estimate());
#else
            local_keyfrm->set_cam_pose(keyfrm_vtx->estimate());
#endif
        }

        for (auto id_local_lm_pair : local_lms) {
            auto local_lm = id_local_lm_pair.second;

            auto lm_vtx = lm_vtx_container.get_vertex(local_lm);
#ifdef USE_HOMOGENEOUS_LANDMARKS
            local_lm->set_pos_in_world(lm_vtx->estimate().hnormalized());
#else
            local_lm->set_pos_in_world(lm_vtx->estimate());
#endif
            local_lm->update_normal_and_depth();
        }
    }

    std::cout<<"LOCAL BUNDLE ADJUSTMENT TOOK: "<<entire_time.ElapsedTimeInSeconds()<<" s.\n";
}

} // namespace optimize
} // namespace openvslam
