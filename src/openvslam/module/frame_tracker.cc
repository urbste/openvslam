#include "openvslam/camera/base.h"
#include "openvslam/data/frame.h"
#include "openvslam/data/keyframe.h"
#include "openvslam/data/landmark.h"
#include "openvslam/match/bow_tree.h"
#include "openvslam/match/projection.h"
#include "openvslam/match/robust.h"
#include "openvslam/match/sparse_feature_aligner.h"
#include "openvslam/match/sparse_image_aligner.h"
#include "openvslam/module/frame_tracker.h"
#include "openvslam/camera/perspective.h"
#include "openvslam/camera/fisheye.h"
#include "openvslam/camera/equirectangular.h"
#include "openvslam/camera/radial_division.h"
#include <spdlog/spdlog.h>
#include <memory>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace openvslam {
namespace module {

void get_undist_pt_and_bearing(const data::frame& curr_frame,
                               const cv::KeyPoint& kp_curr,
                               cv::KeyPoint& undistorted_kp,
                               Vec3_t& bearing) {
    // undistort keypoint and save
    switch (curr_frame.camera_->model_type_) {
    case camera::model_type_t::Perspective: {
        auto c = static_cast<camera::perspective*>(curr_frame.camera_);
        undistorted_kp = c->undistort_keypoint(kp_curr);
        bearing = c->convert_keypoint_to_bearing(undistorted_kp);
    } break;
    case camera::model_type_t::Fisheye: {
        auto c = static_cast<camera::fisheye*>(curr_frame.camera_);
        undistorted_kp = c->undistort_keypoint(kp_curr);
        bearing = c->convert_keypoint_to_bearing(undistorted_kp);
    }break;
    case camera::model_type_t::Equirectangular: {
        auto c = static_cast<camera::equirectangular*>(curr_frame.camera_);
        undistorted_kp = c->undistort_keypoint(kp_curr);
        bearing = c->convert_keypoint_to_bearing(undistorted_kp);
    }break;
    case camera::model_type_t::RadialDivision: {
        auto c = static_cast<camera::radial_division*>(curr_frame.camera_);
        undistorted_kp = c->undistort_keypoint(kp_curr);
        bearing = c->convert_keypoint_to_bearing(undistorted_kp);
    }break;
    }
}

frame_tracker::frame_tracker(camera::base* camera, const unsigned int num_matches_thr)
    : camera_(camera), num_matches_thr_(num_matches_thr), pose_optimizer_() {
    sparse_image_align_.reset(new match::sparse_image_aligner());
}

bool frame_tracker::motion_based_track(data::frame& curr_frm, const data::frame& last_frm, const Mat44_t& velocity) const {
    match::projection projection_matcher(0.9, true);


    if (!curr_frm.features_extracted_) {
        curr_frm.run_feature_extraction();
    }
    // motion modelを使って姿勢の初期値を設定
    curr_frm.set_cam_pose(velocity * last_frm.cam_pose_cw_);

    // 2D-3D対応を初期化
    std::fill(curr_frm.landmarks_.begin(), curr_frm.landmarks_.end(), nullptr);

    // last frameで見えている3次元点を再投影して2D-3D対応を見つける
    const float margin = (camera_->setup_type_ != camera::setup_type_t::Stereo) ? 20 : 10;
    auto num_matches = projection_matcher.match_current_and_last_frames(curr_frm, last_frm, margin);

    if (num_matches < num_matches_thr_) {
        // marginを広げて再探索
        std::fill(curr_frm.landmarks_.begin(), curr_frm.landmarks_.end(), nullptr);
        num_matches = projection_matcher.match_current_and_last_frames(curr_frm, last_frm, 2 * margin);
    }

    if (num_matches < num_matches_thr_) {
        spdlog::debug("motion based tracking failed: {} matches < {}", num_matches, num_matches_thr_);
        return false;
    }

    // pose optimization
    pose_optimizer_.optimize(curr_frm);

    // outlierを除く
    const auto num_valid_matches = discard_outliers(curr_frm);

    if (num_valid_matches < num_matches_thr_) {
        spdlog::debug("motion based tracking failed: {} inlier matches < {}", num_valid_matches, num_matches_thr_);
        return false;
    }
    else {
        return true;
    }
}

bool frame_tracker::bow_match_based_track(data::frame& curr_frm, const data::frame& last_frm, data::keyframe* ref_keyfrm) const {
    match::bow_tree bow_matcher(0.7, true);

    // this can happen in sparse image alignment
    if (!curr_frm.features_extracted_) {
        // first clear all keypoints and features
        curr_frm.run_feature_extraction(true);
    }

    // keyframeとframeで2D対応を探して，frameの特徴点とkeyframeで観測している3次元点の対応を得る
    std::vector<data::landmark*> matched_lms_in_curr;
    auto num_matches = bow_matcher.match_frame_and_keyframe(ref_keyfrm, curr_frm, matched_lms_in_curr);

    if (num_matches < num_matches_thr_) {
        spdlog::debug("bow match based tracking failed: {} matches < {}", num_matches, num_matches_thr_);
        return false;
    }

    // 2D-3D対応情報を更新
    curr_frm.landmarks_ = matched_lms_in_curr;

    // pose optimization
    // 初期値は前のフレームの姿勢
    curr_frm.set_cam_pose(last_frm.cam_pose_cw_);
    pose_optimizer_.optimize(curr_frm);

    // outlierを除く
    const auto num_valid_matches = discard_outliers(curr_frm);

    if (num_valid_matches < num_matches_thr_) {
        spdlog::debug("bow match based tracking failed: {} inlier matches < {}", num_valid_matches, num_matches_thr_);
        return false;
    }
    else {
        return true;
    }
}

bool frame_tracker::robust_match_based_track(data::frame& curr_frm, const data::frame& last_frm, data::keyframe* ref_keyfrm) const {
    match::robust robust_matcher(0.8, false);

    // this can happen in sparse image alignment
    if (!curr_frm.features_extracted_) {
        // first clear all keypoints and features
        curr_frm.run_feature_extraction(true);
    }

    // keyframeとframeで2D対応を探して，frameの特徴点とkeyframeで観測している3次元点の対応を得る
    std::vector<data::landmark*> matched_lms_in_curr;
    auto num_matches = robust_matcher.match_frame_and_keyframe(curr_frm, ref_keyfrm, matched_lms_in_curr);

    if (num_matches < num_matches_thr_) {
        spdlog::debug("robust match based tracking failed: {} matches < {}", num_matches, num_matches_thr_);
        return false;
    }

    // 2D-3D対応情報を更新
    curr_frm.landmarks_ = matched_lms_in_curr;

    // pose optimization
    // 初期値は前のフレームの姿勢
    curr_frm.set_cam_pose(last_frm.cam_pose_cw_);
    pose_optimizer_.optimize(curr_frm);

    // outlierを除く
    const auto num_valid_matches = discard_outliers(curr_frm);

    if (num_valid_matches < num_matches_thr_) {
        spdlog::debug("robust match based tracking failed: {} inlier matches < {}", num_valid_matches, num_matches_thr_);
        return false;
    }
    else {
        return true;
    }
}

bool frame_tracker::sparse_img_alignment_track(
        data::frame &curr_frm,
        const data::frame &last_frm,
        const Mat44_t &velocity) const {

    // motion modelを使って姿勢の初期値を設定
    curr_frm.set_cam_pose(velocity * last_frm.cam_pose_cw_);

    // check if last frame have enough observations
    size_t inliers_in_last_frame = 0;
    for (unsigned int i = 0; i < last_frm.num_keypts_; i++)
        if (last_frm.landmarks_[i] && last_frm.landmarks_[i]->will_be_erased() == false)
            inliers_in_last_frame++;
    if (inliers_in_last_frame < 30) {
        spdlog::info("Last frame have less observations than: {}"
                     " sparse alignment may have a erroneous "
                     "result, return back to feature method.",inliers_in_last_frame);
        return false;
    }

    Mat44_t TCR;

    sparse_image_align_->set_parameters(last_frm.image_pyramid_.size()-1, 1, 10, false, false);
    size_t ret = sparse_image_align_->run(&last_frm, &curr_frm, TCR);

    if (ret < 60) {
        spdlog::info("Sparse feature alignment failed. Falling back to feature methods");
        curr_frm.set_cam_pose(velocity * last_frm.cam_pose_cw_);
        return false;
    }

    curr_frm.set_cam_pose(TCR * last_frm.cam_pose_cw_);

    return true;
}

std::vector<std::pair<data::keyframe *, unsigned int>> SelectNearestKeyframe(
        const std::map<data::keyframe*, unsigned int> &observations,
        data::keyframe* ref_kf, int n) {

    std::vector<std::pair<data::keyframe*, unsigned int> > s;
    for (auto &o: observations) {
        if (!o.first->will_be_erased() && o.first != ref_kf)
            s.push_back(std::make_pair(o.first, o.second));
    }

    std::sort(s.begin(), s.end(),
         [](const std::pair<data::keyframe*, unsigned int> &p1,
            const std::pair<data::keyframe*, unsigned int> &p2) {
             return p1.first->id_ > p2.first->id_;
         });

    if ((int) s.size() < n)
        return s;
    else
        return std::vector<std::pair<data::keyframe *, unsigned int>>(s.begin(), s.begin() + n);
}

double get_convex_hull_area(const std::vector<cv::Point2f>& originalPoints,
                            const double img_area) {
    if (originalPoints.empty()) {
        return img_area;
    }
    std::vector<cv::Point2f> convexHull;  // Convex hull points
    // Calculate convex hull of original points (which points positioned on the boundary)
    cv::convexHull(originalPoints,convexHull,false);

    const double area = fabs(cv::contourArea(convexHull));

    return area / img_area * 100.0;
}

unsigned int frame_tracker::sparse_feat_alignment_track(
        data::frame& curr_frame,
        data::keyframe* ref_keyframe,
        std::vector<data::landmark*> local_landmarks,
        std::set<data::landmark*>& direct_map_points_cache) const {

    optimize::pose_optimizer sparse_feat_pose_optimizer_(1, 10);


    int cntSuccess = 0;
    // use grid to evaluate the coverage of feature points
    const int grid_size = 5;
    const int grid_rows = curr_frame.image_pyramid_[0].rows / grid_size;
    const int grid_cols = curr_frame.image_pyramid_[0].cols / grid_size;
    std::vector<bool> grid(grid_rows * grid_cols, false);
    std::vector<cv::Point2f> convex_hull_pts;
    if (!direct_map_points_cache.empty()) {
        for (auto iter = direct_map_points_cache.begin(); iter != direct_map_points_cache.end();) {
            data::landmark *mp = *iter;
            // first check if there is a landmark
            if (mp->will_be_erased() || mp == nullptr) {
                iter = direct_map_points_cache.erase(iter);
                continue;
            }

            // projection in current frame
            Vec2_t track_xy;
            unsigned int pred_scale_level;
            float x_right;
            if (!curr_frame.can_observe(mp, 0.5, track_xy, x_right, pred_scale_level, true)) {
                iter = direct_map_points_cache.erase(iter);
                continue;
            }

            int gx = static_cast<int> (track_xy[0] / grid_size );
            int gy = static_cast<int> (track_xy[1] / grid_size );
            int k = gy * grid_cols + gx;

            if (grid[k] == true) {
                iter++;
                continue;        // already exist a projection
            }

            match::sparse_feature_aligner sparse_feat_aligner;
            // try align it with current frame
            const std::map<data::keyframe*, unsigned int> obs = mp->get_observations();
            auto obs_sorted = SelectNearestKeyframe(obs, ref_keyframe, 5);
            eigen_alloc_vector<Vec2_t> matched_pixels;
            for (auto o: obs_sorted) {
                int level = mp->scale_level_in_tracking_;
                Vec2_t px_curr(track_xy[0],track_xy[1]);
                if (sparse_feat_aligner.find_projection_direct(o.first, &curr_frame,
                                                               mp, px_curr, level)) {
                    if (px_curr[0] < 20 || px_curr[1] < 20
                        || px_curr[0] >= curr_frame.image_pyramid_[0].cols - 20
                        || px_curr[1] >= curr_frame.image_pyramid_[0].rows - 20)
                        continue;
                    matched_pixels.push_back(px_curr);
                    break;
                }
            }
            if (!matched_pixels.empty()) {
                Vec2_t px_ave(0, 0);
                for (Vec2_t &p: matched_pixels)
                    px_ave += p;
                px_ave = px_ave / matched_pixels.size();

                const cv::KeyPoint kp_curr = cv::KeyPoint(
                            cv::Point2f(px_ave[0], px_ave[1]), 7, -1, 0, 0);
                curr_frame.keypts_.push_back(kp_curr);
                // undistort keypoint and save
                cv::KeyPoint undist_kp;
                Vec3_t bearing;
                get_undist_pt_and_bearing(curr_frame, kp_curr, undist_kp, bearing);
                curr_frame.undist_keypts_.push_back(undist_kp);
                curr_frame.bearings_.push_back(bearing);
                curr_frame.landmarks_.push_back(mp);
                curr_frame.depths_.push_back(-1);
                curr_frame.outlier_flags_.push_back(false);
                curr_frame.stereo_x_right_.push_back(-1.f);
                convex_hull_pts.push_back(cv::Point2f(kp_curr.pt.x, kp_curr.pt.y));
                int gx = static_cast<int> ( px_ave[0] / grid_size );
                int gy = static_cast<int> ( px_ave[1] / grid_size );
                int k = gy * grid_cols + gx;
                grid[k] = true;

                iter++;
                cntSuccess++;
            } else {
                iter = direct_map_points_cache.erase(iter);
            }
        }
    }
    const double img_area =
            curr_frame.image_pyramid_[0].rows *
            curr_frame.image_pyramid_[0].cols;
    const double percent_occupied = get_convex_hull_area(convex_hull_pts, img_area);
    std::cout<<"img pts occupy "<<percent_occupied<<" percent of image area\n";
    if (percent_occupied > 50.0 && !direct_map_points_cache.empty()) {
        // we matched enough points in cache, then do pose optimization
        curr_frame.num_keypts_ = curr_frame.keypts_.size();
        curr_frame.stereo_x_right_.resize(curr_frame.num_keypts_,-1.0f);
        sparse_feat_pose_optimizer_.optimize(curr_frame);
        unsigned int num_valid_pts = discard_outliers_sparse(curr_frame, direct_map_points_cache);
        if (num_valid_pts < 50) {
            spdlog::warn("re Track Local Map direct failed");
            return num_valid_pts;
        } else {
            return num_valid_pts;
        }
    }
    spdlog::info("Searching more landmarks for direct tracking");

    int rejected = 0;
    int outside = 0;
    for (data::landmark* mp : local_landmarks) {
        if (direct_map_points_cache.find(mp) != direct_map_points_cache.end()) {
            continue;
        }
        if (mp->will_be_erased())
            continue;
        Vec2_t track_xy;
        unsigned int pred_scale_level;
        float x_right;

        if (!curr_frame.can_observe(mp, 0.5, track_xy, x_right, pred_scale_level, true))
        {
            outside++;
            continue;
        }

//        int gx = static_cast<int> (track_xy[0] / grid_size );
//        int gy = static_cast<int> (track_xy[1] / grid_size );
//        int k = gy * grid_cols + gx;

//        if (grid[k] == true) {
//            continue;        // already exist a projection in that grid
//        }

        match::sparse_feature_aligner sparse_feat_aligner;
        // try align it with current frame
        const std::map<data::keyframe*, unsigned int> obs = mp->get_observations();
        auto obs_sorted = SelectNearestKeyframe(obs, ref_keyframe, 5);
        eigen_alloc_vector<Vec2_t> matched_pixels;
        for (auto o: obs_sorted) {
            int level = mp->scale_level_in_tracking_;
            Vec2_t px_curr(track_xy[0],track_xy[1]);
            if (sparse_feat_aligner.find_projection_direct(o.first, &curr_frame,
                                                           mp, px_curr, level)) {
                if (px_curr[0] < 20 || px_curr[1] < 20
                    || px_curr[0] >= curr_frame.image_pyramid_[0].cols - 20
                    || px_curr[1] >= curr_frame.image_pyramid_[0].rows - 20)
                    continue;
                matched_pixels.push_back(px_curr);
                break;
            }
        }

        if (!matched_pixels.empty()) {
            Vec2_t px_ave(0, 0);
            for (Vec2_t &p: matched_pixels)
                px_ave += p;
            px_ave = px_ave / matched_pixels.size();



            // insert a feature and assign it to a map point
            const cv::KeyPoint kp_curr = cv::KeyPoint(cv::Point2f(px_ave[0], px_ave[1]), 7, -1, 0, 0);
            curr_frame.keypts_.push_back(kp_curr);
            cv::KeyPoint undist_kp;
            Vec3_t bearing;
            get_undist_pt_and_bearing(curr_frame, kp_curr, undist_kp, bearing);
            curr_frame.undist_keypts_.push_back(undist_kp);
            curr_frame.bearings_.push_back(bearing);
            curr_frame.landmarks_.push_back(mp);
            curr_frame.depths_.push_back(-1);
            curr_frame.outlier_flags_.push_back(false);
            curr_frame.stereo_x_right_.push_back(-1.f);

//            int gx = static_cast<int> ( px_ave[0] / grid_size );
//            int gy = static_cast<int> ( px_ave[1] / grid_size );
//            int k = gy * grid_cols + gx;
//            grid[k] = true;

            direct_map_points_cache.insert(mp);
        } else {
            rejected++;
        }
    }

    curr_frame.num_keypts_ = curr_frame.keypts_.size();
    curr_frame.keypts_right_.resize(curr_frame.num_keypts_);
    sparse_feat_pose_optimizer_.optimize(curr_frame);
    unsigned int num_valid_pts = discard_outliers_sparse(curr_frame, direct_map_points_cache);
    if (num_valid_pts < 50) {
        spdlog::warn("Track Local Map direct failed");
        return num_valid_pts;
    } else {
        return num_valid_pts;
    }
}

unsigned int frame_tracker::discard_outliers_sparse(
        data::frame& curr_frm,
        std::set<data::landmark*>& direct_map_points_cache) const {

    unsigned int num_valid_matches = 0;

    for (unsigned int idx = 0; idx < curr_frm.num_keypts_; ++idx) {
        if (!curr_frm.landmarks_.at(idx)) {
            continue;
        }

        auto lm = curr_frm.landmarks_.at(idx);

        if (curr_frm.outlier_flags_.at(idx)) {
            curr_frm.landmarks_.at(idx) = nullptr;
            curr_frm.outlier_flags_.at(idx) = false;
            lm->is_observable_in_tracking_ = false;
            lm->identifier_in_local_lm_search_ = curr_frm.id_;
            //auto iter = direct_map_points_cache.find(curr_frm.landmarks_[idx]);
            //if (iter != direct_map_points_cache.end())
            //    direct_map_points_cache.erase(iter);
        }
        else {
            //curr_frm.outlier_flags_.at(idx) = false;
            curr_frm.landmarks_[idx]->increase_num_observed();
             ++num_valid_matches;
        }
    }

    return num_valid_matches;
}

unsigned int frame_tracker::discard_outliers(data::frame& curr_frm) const {
    unsigned int num_valid_matches = 0;

    for (unsigned int idx = 0; idx < curr_frm.num_keypts_; ++idx) {
        if (!curr_frm.landmarks_.at(idx)) {
            continue;
        }

        auto lm = curr_frm.landmarks_.at(idx);

        if (curr_frm.outlier_flags_.at(idx)) {
            curr_frm.landmarks_.at(idx) = nullptr;
            curr_frm.outlier_flags_.at(idx) = false;
            lm->is_observable_in_tracking_ = false;
            lm->identifier_in_local_lm_search_ = curr_frm.id_;
            continue;
        }

        ++num_valid_matches;
    }

    return num_valid_matches;
}

} // namespace module
} // namespace openvslam
