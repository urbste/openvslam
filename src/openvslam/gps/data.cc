// created by Steffen Urban April 2020, urbste@gmail.com
#include "openvslam/gps/data.h"
#include "openvslam/util/gps_converter.h"

namespace openvslam {
namespace gps {

data::data(const double latitude, const double longitude, const double height,
           const double dop_precision, const int fix, const int speed_2d, const int speed_3d,
           const double ts)
    : llh_(latitude, longitude, height), dop_precision_(dop_precision),
      speed_2d_(speed_2d), speed_3d_(speed_3d), fix_(fix), ts_(ts) {
    xyz_ = openvslam::util::gps_converter::LLAToECEF(llh_);
}

data::data(const Vec3_t& llh,  const double dop_precision,
           const int fix, const double speed_2d, const double speed_3d,
           const double ts)
    : llh_(llh), dop_precision_(dop_precision),
      speed_2d_(speed_2d), speed_3d_(speed_3d), fix_(fix), ts_(ts) {
    xyz_ = openvslam::util::gps_converter::LLAToECEF(llh_);
}

} // namespace gps
} // namespace openvslam
