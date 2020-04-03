#ifndef OPENVSLAM_OPTIMIZER_G2O_GPS_PRIOR_EDGE_H
#define OPENVSLAM_OPTIMIZER_G2O_GPS_PRIOR_EDGE_H

// implemented by Steffen Urban March 2020 (urbste@googlemail.com)
// basically taken from
// https://github.com/introlab/rtabmap/blob/master/corelib/src/optimizer/g2o/edge_se3_xyzprior.h
#include "openvslam/type.h"

#include <g2o/core/base_vertex.h>
#include <g2o/types/slam3d/se3quat.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/types/slam3d/parameter_se3_offset.h>

namespace openvslam {
namespace optimize {
namespace g2o {

//class gps_prior_edge final : public ::g2o::BaseUnaryEdge<3, Eigen::Vector3d, ::g2o::SE3Quat> {

//    public:
//      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
//      gps_prior_edge();

//      virtual void setMeasurement(const Eigen::Vector3d& m) {
//        _measurement = m;
//      }

//      virtual bool setMeasurementData(const double * d) {
//        Eigen::Map<const Eigen::Vector3d> v(d);
//        _measurement = v;
//        return true;
//      }

//      virtual bool getMeasurementData(double* d) const {
//        Eigen::Map<Eigen::Vector3d> v(d);
//        v = _measurement;
//        return true;
//      }

//      virtual int measurementDimension() const {return 3;}

//      virtual bool read(std::istream& is);
//      virtual bool write(std::ostream& os) const;
//      virtual void computeError();
//      virtual bool setMeasurementFromState();

//      virtual double initialEstimatePossible(const g2o::OptimizableGraph::VertexSet& /*from*/, g2o::OptimizableGraph::Vertex* /*to*/) {return 1.;}
//      virtual void initialEstimate(const g2o::OptimizableGraph::VertexSet& /*from_*/, g2o::OptimizableGraph::Vertex* /*to_*/);

//      const g2o::ParameterSE3Offset* offsetParameter() { return _offsetParam; }

//    protected:
//      virtual bool resolveCaches();
//      g2o::ParameterSE3Offset* _offsetParam;
//      g2o::CacheSE3Offset* _cache;

//};

} // namespace g2o
} // namespace optimize
} // namespace openvslam

#endif // OPENVSLAM_OPTIMIZER_G2O_GPS_PRIOR_EDGE_H
