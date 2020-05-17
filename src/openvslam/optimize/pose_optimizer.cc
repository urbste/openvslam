#include "openvslam/data/frame.h"
#include "openvslam/data/landmark.h"
#include "openvslam/optimize/pose_optimizer.h"
#include "openvslam/optimize/g2o/se3/pose_opt_edge_wrapper.h"
#include "openvslam/util/converter.h"
#include "openvslam/optimize/g2o/se3/SE3hom.h"
#include "openvslam/util/timer.h"

#include <vector>
#include <mutex>

#include <Eigen/StdVector>
#include <g2o/core/solver.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/sparse_optimizer_terminate_action.h>

namespace openvslam {
namespace optimize {

//void jac_test(Vec6_t& min_vec,
//              Vec4_t& pos_w_hom_,
//              Eigen::Matrix<double, 2, 6>& _jacobianOplusXi) {
//    Mat33_t R;
//    g2o::se3::rotation_from_rodr_vec(min_vec.tail(3), R);
//    // project point to cam and normalize to create new observation
//    // this will create x_i^a in eq. 12.225
//    Vec3_t x_i_a;
//    g2o::se3::project_hom_point_to_camera(min_vec, pos_w_hom_, x_i_a);

//    // now we need the nullspace and the jacobian of that point
//    nullspace32_t ns_x_i_a;
//    Mat33_t jac_x_i_a;
//    g2o::nullS_3x2_templated<double>(x_i_a, ns_x_i_a);
//    g2o::jacobian_3_vec<double>(x_i_a, jac_x_i_a);

//    Eigen::Matrix<double, 2, 3> ns_js = ns_x_i_a.transpose() * jac_x_i_a;
//    _jacobianOplusXi.block<2,3>(0,0) = ns_js * (pos_w_hom_(3) * R);
//    Vec3_t tmp1 = R * (pos_w_hom_.head(3) - pos_w_hom_(3)*min_vec.head(3));
//    Mat33_t skew_;
//    g2o::se3::skew_mat(tmp1, skew_);
//    _jacobianOplusXi.block<2,3>(0,3) = ns_js*skew_;
//}




//void evaluateJacobian(g2o::se3::mono_perspective_pose_opt_edge_hom* e,
//                           ::g2o::JacobianWorkspace& numericJacobianWorkspace)
//{
//  // calling the analytic Jacobian but writing to the numeric workspace
//  e->linearizeOplus(numericJacobianWorkspace);
////  // copy result into analytic workspace
////  jacobianWorkspace = numericJacobianWorkspace;

////  // compute the numeric Jacobian into the numericJacobianWorkspace workspace as setup by the previous call
//  Eigen::Matrix<double, 2, 6, Eigen::RowMajor> jac;
//  e->linearizeOplus_AUTODIFF(jac);

//  // compare the Jacobians
//  double* n = numericJacobianWorkspace.workspaceForVertex(0);
//  //numElems *= EdgeType1::VertexXiType::Dimension;
//  Eigen::Map<Eigen::Matrix<double, 2, 6>> jac_num(n);
//  std::cout<<"jac_num: \n"<<jac_num<<std::endl;

//  std::cout<<"jac: \n"<<jac<<std::endl;
//  //int numElems = EdgeType1::Dimension;


//}

pose_optimizer::pose_optimizer(const unsigned int num_trials, const unsigned int num_each_iter)
    : num_trials_(num_trials), num_each_iter_(num_each_iter) {}

unsigned int pose_optimizer::optimize(data::frame& frm) const {
    util::Timer entire_time;

    // 1. optimizerを構築

    auto linear_solver = ::g2o::make_unique<::g2o::LinearSolverDense<::g2o::BlockSolver_6_3::PoseMatrixType>>();
    auto block_solver = ::g2o::make_unique<::g2o::BlockSolver_6_3>(std::move(linear_solver));
    auto algorithm = new ::g2o::OptimizationAlgorithmLevenberg(std::move(block_solver));

    ::g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(algorithm);
    //optimizer.setVerbose(true);
    unsigned int num_init_obs = 0;

    // 2. frameをg2oのvertexに変換してoptimizerにセットする

#ifdef USE_HOMOGENEOUS_LANDMARKS
    auto frm_vtx = new g2o::se3::shot_vertex_hom();
    Vec6_t cam_pose_init;
    g2o::se3::world_2_cam_trafo_to_vec_6(frm.cam_pose_cw_, cam_pose_init);
    frm_vtx->setEstimate(cam_pose_init);
#else
    auto frm_vtx = new g2o::se3::shot_vertex();
    frm_vtx->setEstimate(util::converter::to_g2o_SE3(frm.cam_pose_cw_));
#endif

    frm_vtx->setId(frm.id_);
    frm_vtx->setFixed(false);
    optimizer.addVertex(frm_vtx);

    const unsigned int num_keypts = frm.num_keypts_;

    // 3. landmarkのvertexをreprojection edgeで接続する

    // reprojection edgeのcontainer
    using pose_opt_edge_wrapper = g2o::se3::pose_opt_edge_wrapper<data::frame>;
    std::vector<pose_opt_edge_wrapper> pose_opt_edge_wraps;
    pose_opt_edge_wraps.reserve(num_keypts);

    constexpr float chi_sq_2D = 5.99146;
    constexpr float chi_sq_3D = 7.81473;

    const float sqrt_chi_sq_2D = std::sqrt(chi_sq_2D);
    const float sqrt_chi_sq_3D = std::sqrt(chi_sq_3D);

    // just one edge to check jacs
    ::g2o::JacobianWorkspace numericJacobianWorkspace;


    for (unsigned int idx = 0; idx < num_keypts; ++idx) {
        auto lm = frm.landmarks_.at(idx);
        if (!lm) {
            continue;
        }
        if (lm->will_be_erased()) {
            continue;
        }

        ++num_init_obs;
        frm.outlier_flags_.at(idx) = false;

        // frameのvertexをreprojection edgeで接続する
        const auto& undist_keypt = frm.undist_keypts_.at(idx);
        const float x_right = frm.stereo_x_right_.at(idx);
        const float inv_sigma_sq = frm.inv_level_sigma_sq_.at(undist_keypt.octave);
        const auto sqrt_chi_sq = (frm.camera_->setup_type_ == camera::setup_type_t::Monocular)
                                     ? sqrt_chi_sq_2D
                                     : sqrt_chi_sq_3D;
#ifdef USE_HOMOGENEOUS_LANDMARKS
        const auto& bearing = frm.bearings_.at(idx);
        const auto& jac = frm.bearings_jac_.at(idx);
        const auto& nullspace = frm.bearings_nullspace_.at(idx);
        auto pose_opt_edge_wrap = pose_opt_edge_wrapper(&frm, frm_vtx, lm->get_pos_in_world(),
                                                        idx, undist_keypt.pt.x, undist_keypt.pt.y, x_right,
                                                        inv_sigma_sq, sqrt_chi_sq,
                                                        bearing, jac, nullspace);
#else
        auto pose_opt_edge_wrap = pose_opt_edge_wrapper(&frm, frm_vtx, lm->get_pos_in_world(),
                                                        idx, undist_keypt.pt.x, undist_keypt.pt.y, x_right,
                                                        inv_sigma_sq, sqrt_chi_sq);
#endif
        pose_opt_edge_wraps.push_back(pose_opt_edge_wrap);
        optimizer.addEdge(pose_opt_edge_wrap.edge_);
        numericJacobianWorkspace.updateSize(pose_opt_edge_wrap.edge_);
        numericJacobianWorkspace.allocate();

        //evaluateJacobian(static_cast<g2o::se3::mono_perspective_pose_opt_edge_hom*>(
        //                          pose_opt_edge_wrap.edge_),numericJacobianWorkspace);
    }


    if (num_init_obs < 5) {
        return 0;
    }

    // 4. robust BAを実行する
    ::g2o::SparseOptimizerTerminateAction* terminateAction = 0;
        terminateAction = new ::g2o::SparseOptimizerTerminateAction;
     terminateAction->setGainThreshold(1e-6);
    optimizer.addPostIterationAction(terminateAction);

    unsigned int num_bad_obs = 0;
    for (unsigned int trial = 0; trial < num_trials_; ++trial) {
        optimizer.initializeOptimization();
        optimizer.optimize(num_each_iter_);

        num_bad_obs = 0;

        for (auto& pose_opt_edge_wrap : pose_opt_edge_wraps) {
            auto edge = pose_opt_edge_wrap.edge_;

            if (frm.outlier_flags_.at(pose_opt_edge_wrap.idx_)) {
                edge->computeError();
            }

            if (pose_opt_edge_wrap.is_monocular_) {
                if (chi_sq_2D < edge->chi2()) {
                    frm.outlier_flags_.at(pose_opt_edge_wrap.idx_) = true;
                    pose_opt_edge_wrap.set_as_outlier();
                    ++num_bad_obs;
                }
                else {
                    frm.outlier_flags_.at(pose_opt_edge_wrap.idx_) = false;
                    pose_opt_edge_wrap.set_as_inlier();
                }
            }
            else {
                if (chi_sq_3D < edge->chi2()) {
                    frm.outlier_flags_.at(pose_opt_edge_wrap.idx_) = true;
                    pose_opt_edge_wrap.set_as_outlier();
                    ++num_bad_obs;
                }
                else {
                    frm.outlier_flags_.at(pose_opt_edge_wrap.idx_) = false;
                    pose_opt_edge_wrap.set_as_inlier();
                }
            }

            if (trial == num_trials_ - 2) {
                edge->setRobustKernel(nullptr);
            }
        }

        if (num_init_obs - num_bad_obs < 5) {
            break;
        }
    }

    // 5. 情報を更新
#ifdef USE_HOMOGENEOUS_LANDMARKS
    frm.set_cam_pose_min(frm_vtx->estimate());
#else
    frm.set_cam_pose(frm_vtx->estimate());
#endif
    //std::cout<<"Optimizing the pose took: "<<entire_time.ElapsedTimeInSeconds()<<" s"<<std::endl;
    return num_init_obs - num_bad_obs;
}

} // namespace optimize
} // namespace openvslam
