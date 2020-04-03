#include "openvslam/optimize/g2o/se3/gps_prior_edge.h"

#include <Eigen/Core>
#include <g2o/types/slam3d/se3quat.h>
#include <g2o/core/base_unary_edge.h>
namespace openvslam {
namespace optimize {
namespace g2o {


//gps_prior_edge::gps_prior_edge() : BaseUnaryEdge<3, Eigen::Vector3d, ::g2o::SE3Quat>()
//{
//  information().setIdentity();
//  setMeasurement(Vector3::Zero());
//  _cache = 0;
//  _offsetParam = 0;
//  resizeParameters(1);
//  installParameter(_offsetParam, 0);
//}

//bool gps_prior_edge::resolveCaches(){
//  assert(_offsetParam);
//  ::g2o::ParameterVector pv(1);
//  pv[0] = _offsetParam;
//  resolveCache(_cache, (OptimizableGraph::Vertex*)_vertices[0], "CACHE_SE3_OFFSET", pv);
//  return _cache != 0;
//}

//bool gps_prior_edge::read(std::istream& is)
//{
//  int pid;
//  is >> pid;
//  if (!setParameterId(0, pid))
//    return false;

//  // measured keypoint
//  ::g2o::Vector3 meas;
//  for (int i = 0; i < 3; i++) is >> meas[i];
//  setMeasurement(meas);

//  // read covariance matrix (upper triangle)
//  if (is.good()) {
//    for (int i = 0; i < 3; i++) {
//      for (int j = i; j < 3; j++) {
//        is >> information()(i,j);
//        if (i != j)
//          information()(j,i) = information()(i,j);
//      }
//    }
//  }
//  return !is.fail();
//}

//bool gps_prior_edge::write(std::ostream& os) const {
//  os << _offsetParam->id() <<  " ";
//  for (int i = 0; i < 3; i++) os << measurement()[i] << " ";
//  for (int i = 0; i < 3; i++) {
//    for (int j = i; j < 3; j++) {
//      os << information()(i,j) << " ";
//    }
//  }
//  return os.good();
//}

//void gps_prior_edge::computeError() {
//  const ::g2o::SE3Quat* v = static_cast<const ::g2o::SE3Quat*>(_vertices[0]);
//  _error = v->translation() - _measurement;
//}

//void gps_prior_edge::linearizeOplus() {
//  _jacobianOplusXi << ::g2o::Matrix3::Identity();
//}

//bool gps_prior_edge::setMeasurementFromState() {
//  const ::g2o::SE3Quat* v = static_cast<const ::g2o::SE3Quat*>(_vertices[0]);
//  _measurement = v->translation();
//  return true;
//}

//void gps_prior_edge::initialEstimate(const OptimizableGraph::VertexSet& /*from_*/,
//                                     OptimizableGraph::Vertex* /*to_*/) {
//  ::g2o::SE3Quat *v = static_cast<::g2o::SE3Quat*>(_vertices[0]);
//  assert(v && "Vertex for the Prior edge is not set");

//  ::g2o::Isometry3 newEstimate = _offsetParam->offset().inverse() * Eigen::Translation3d(measurement());
//  if (_information.block<3,3>(0,0).array().abs().sum() == 0){ // do not set translation, as that part of the information is all zero
//    newEstimate.translation() = v->translation();
//  }
//  v->setEstimate(newEstimate);
//}

} // namespace g2o
} // namespace optimize
} // namespace openvslam
