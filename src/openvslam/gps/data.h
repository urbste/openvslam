#ifndef OPENVSLAM_GPS_DATA_H
#define OPENVSLAM_GPS_DATA_H

#include "openvslam/type.h"

namespace openvslam {
namespace gps {

class data {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    //! default constructor
    data() {}
    //! Constructor for scalar inputs
    data(const double latitude, const double longitude, const double height,
         const double dop_precision, const int fix, const int speed_2d, const int speed_3d,
         const double ts);

    //! Constructor for vector inputs
    data(const Vec3_t& llh, const double dop_precision, const int fix,
         const double speed_2d, const double speed_3d, const double ts);

    //! gps measurement in latitude, longitude and height
    Vec3_t llh_;
    //! gps measurement in x y z ellipsoid coordinates
    Vec3_t xyz_;
    //! dilusion of precision
    double dop_precision_;
    //! 2D speed
    double speed_2d_;
    //! 3D speed
    double speed_3d_;
    //! fix -> 0: no fix, 1: 2D fix, 2: 3D fix
    int fix_;
    //! timestamp [s]
    double ts_;
};

} // namespace gps
} // namespace openvslam

#endif // OPENVSLAM_GPS_DATA_H
