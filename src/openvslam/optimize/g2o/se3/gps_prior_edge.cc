#include "openvslam/optimize/g2o/se3/gps_prior_edge.h"

#include <Eigen/Core>
#include <g2o/types/slam3d/se3quat.h>
#include <g2o/core/base_unary_edge.h>
namespace openvslam {
namespace optimize {
namespace g2o {
namespace se3 {


gps_prior_edge::gps_prior_edge() : ::g2o::BaseUnaryEdge<3, Vec3_t, shot_vertex>()
{
  information().setIdentity();
  setMeasurement(Vec3_t::Zero());
  _cache = 0;
  _offsetParam = 0;
  resizeParameters(1);
  installParameter(_offsetParam, 0);
}

bool gps_prior_edge::resolveCaches(){
  assert(_offsetParam);
  ::g2o::ParameterVector pv(1);
  pv[0] = _offsetParam;
  resolveCache(_cache, (::g2o::OptimizableGraph::Vertex*)_vertices[0], "CACHE_SE3_OFFSET", pv);
  return _cache != 0;
}

bool gps_prior_edge::read(std::istream& is)
{
  int pid;
  is >> pid;
  if (!setParameterId(0, pid))
    return false;

  // measured keypoint
  Vec3_t meas;
  for (int i = 0; i < 3; i++) is >> meas[i];
  setMeasurement(meas);

  // read covariance matrix (upper triangle)
  if (is.good()) {
    for (int i = 0; i < 3; i++) {
      for (int j = i; j < 3; j++) {
        is >> information()(i,j);
        if (i != j)
          information()(j,i) = information()(i,j);
      }
    }
  }
  return !is.fail();
}

bool gps_prior_edge::write(std::ostream& os) const {
  os << _offsetParam->id() <<  " ";
  for (int i = 0; i < 3; i++) os << measurement()[i] << " ";
  for (int i = 0; i < 3; i++) {
    for (int j = i; j < 3; j++) {
      os << information()(i,j) << " ";
    }
  }
  return os.good();
}

void gps_prior_edge::computeError() {
  const shot_vertex* v = static_cast<const shot_vertex*>(_vertices[0]);
  const Vec3_t cam_center = -v->estimate().rotation().toRotationMatrix()*v->estimate().translation();
  _error = cam_center - _measurement;
}

//bool gps_prior_edge::setMeasurementFromState() {
//  const shot_vertex* v = static_cast<const shot_vertex*>(_vertices[0]);
//  _measurement = v->estimate().translation();
//  return true;
//}

void gps_prior_edge::linearizeOplus() {
   _jacobianOplusXi << Mat33_t::Identity();
 }

//void gps_prior_edge::initialEstimate(const ::g2o::OptimizableGraph::VertexSet& /*from_*/,
//                                     ::g2o::OptimizableGraph::Vertex* /*to_*/) {
//    shot_vertex *v = static_cast<shot_vertex*>(_vertices[0]);
//    assert(v && "Vertex for the Prior edge is not set");

//    ::g2o::Isometry3 newEstimate = _offsetParam->offset().inverse() * Eigen::Translation3d(measurement());
//    if (_information.block<3,3>(0,0).array().abs().sum() == 0){ // do not set translation, as that part of the information is all zero
//      newEstimate.translation() = v->estimate().translation();
//    }
//    v->setEstimate(newEstimate.);
//}

} // namespace se3
} // namespace g2o
} // namespace optimize
} // namespace openvslam
