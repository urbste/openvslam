#ifndef OPENVSLAM_MODULE_FRAME_TRACKER_H
#define OPENVSLAM_MODULE_FRAME_TRACKER_H

#include "openvslam/type.h"
#include "openvslam/optimize/pose_optimizer.h"
#include "openvslam/match/sparse_image_aligner.h"

namespace openvslam {

namespace camera {
class base;
} // namespace camera

namespace data {
class frame;
class keyframe;
class landmark;
} // namespace data

namespace module {

class frame_tracker {
public:
    explicit frame_tracker(camera::base* camera, const unsigned int num_matches_thr = 20);

    bool motion_based_track(data::frame& curr_frm, const data::frame& last_frm, const Mat44_t& velocity) const;

    bool bow_match_based_track(data::frame& curr_frm, const data::frame& last_frm, data::keyframe* ref_keyfrm) const;

    bool robust_match_based_track(data::frame& curr_frm, const data::frame& last_frm, data::keyframe* ref_keyfrm) const;

    bool sparse_img_alignment_track(
            data::frame& curr_frm, const data::frame &last_frm,
            const Mat44_t& velocity) const;

    unsigned int sparse_feat_alignment_track(data::frame& curr_frame,
            data::keyframe* ref_keyframe,
            std::vector<data::landmark *> local_landmarks,
            std::set<data::landmark*>& direct_map_points_cache) const;
private:
    unsigned int discard_outliers(data::frame& curr_frm) const;

    unsigned int discard_outliers_sparse(data::frame& curr_frm,
            std::set<data::landmark*>& direct_map_points_cache) const;

    const camera::base* camera_;
    const unsigned int num_matches_thr_;

    const optimize::pose_optimizer pose_optimizer_;

    int cache_hit_thresh_ = 200;

    std::unique_ptr<match::sparse_image_aligner> sparse_image_align_;
};

} // namespace module
} // namespace openvslam

#endif // OPENVSLAM_MODULE_FRAME_TRACKER_H
