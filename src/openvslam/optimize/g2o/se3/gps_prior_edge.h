#ifndef OPENVSLAM_OPTIMIZER_G2O_GPS_PRIOR_EDGE_H
#define OPENVSLAM_OPTIMIZER_G2O_GPS_PRIOR_EDGE_H

// implemented by Steffen Urban March 2020 (urbste@googlemail.com)
// see also discussion here: https://github.com/introlab/rtabmap/issues/345
#include "openvslam/type.h"
#include "openvslam/optimize/g2o/se3/shot_vertex.h"
#include <g2o/core/base_vertex.h>
#include <g2o/types/slam3d/vertex_se3.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/types/slam3d/parameter_se3_offset.h>

namespace openvslam {
namespace optimize {
namespace g2o {
namespace se3 {

class gps_prior_edge final : public ::g2o::BaseUnaryEdge<3, Vec3_t, shot_vertex> {

    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
      gps_prior_edge();

      virtual void setMeasurement(const Vec3_t& m) {
        _measurement = m;
      }

      virtual bool setMeasurementData(const double * d) {
        Eigen::Map<const Vec3_t> v(d);
        _measurement = v;
        return true;
      }

      virtual bool getMeasurementData(double* d) const {
        Eigen::Map<Vec3_t> v(d);
        v = _measurement;
        return true;
      }

      virtual int measurementDimension() const {return 3;}

      virtual bool read(std::istream& is);
      virtual bool write(std::ostream& os) const;
      virtual void computeError();
      //virtual void linearizeOplus(); // maybe remove that guy?
      //virtual bool setMeasurementFromState();

      //virtual double initialEstimatePossible(const ::g2o::OptimizableGraph::VertexSet& /*from*/, ::g2o::OptimizableGraph::Vertex* /*to*/) {return 1.;}
      //virtual void initialEstimate(const ::g2o::OptimizableGraph::VertexSet& /*from_*/, ::g2o::OptimizableGraph::Vertex* /*to_*/);

      const ::g2o::ParameterSE3Offset* offsetParameter() { return _offsetParam; }

    protected:
      virtual bool resolveCaches();
      ::g2o::ParameterSE3Offset* _offsetParam;
      ::g2o::CacheSE3Offset* _cache;

};

} // namespace se3
} // namespace g2o
} // namespace optimize
} // namespace openvslam

#endif // OPENVSLAM_OPTIMIZER_G2O_GPS_PRIOR_EDGE_H
