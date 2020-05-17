#ifndef OPENVSLAM_OPTIMIZER_G2O_NULLSPACE_UPDATER_H
#define OPENVSLAM_OPTIMIZER_G2O_NULLSPACE_UPDATER_H

#include <Eigen/StdVector>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <iostream>
#include "g2o/core/base_vertex.h"
#include <g2o/core/hyper_graph_action.h>
#include <g2o/core/parameter.h>

#include "landmark_vertex4.h"
#include "landmark_vertex4_container.h"

using namespace std;

namespace openvslam {
namespace optimize {
namespace g2o {


//class UpdateNullSpace : public ::g2o::HyperGraphAction
//{
//  public:
//    UpdateNullSpace(landmark_vertex4_container* lm_container,
//                    std::unordered_map<unsigned int, data::landmark*>& local_lms,
//                    std::vector<size_t> local_landmark_ids)
//    : lm_container_(lm_container), local_lms_(local_lms), lm_indices_in_optimizer_(local_landmark_ids)
//    {  }

//    /// This function is called by the optimizer at the appropriate time
//    virtual ::g2o::HyperGraphAction* operator()(const ::g2o::HyperGraph* graph,
//                                                Parameters* parameters = 0)
//    {
////        for (auto id_local_lm_pair : local_lms_) {
////            auto local_lm = id_local_lm_pair.second;
////            landmark_vertex4* lm = lm_container_->get_vertex(local_lm);
////            if (lm)
////                if (!lm->fixed())
////                    lm->updateNullSpace();
////        }
//        for (auto lm_id : lm_indices_in_optimizer_)
//        {
//            landmark_vertex4* point
//                = const_cast<landmark_vertex4*>(
//                        reinterpret_cast<const landmark_vertex4*>(
//                            graph->vertex(lm_id)));
//            if (point)
//                if (!point->fixed())
//                    point->updateNullSpace();
//        }

//        return this;
//    }

//  protected:
//    std::vector<size_t> lm_indices_in_optimizer_;
//    std::unordered_map<unsigned int, data::landmark*> local_lms_;
//    landmark_vertex4_container* lm_container_;
//};

}
}
}
#endif // OPENVSLAM_OPTIMIZER_G2O_NULLSPACE_UPDATER_H
