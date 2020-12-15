// created by Steffen Urban April 2020, urbste@gmail.com
#include "openvslam/gps/data.h"
#include "openvslam/util/gps_converter.h"

namespace openvslam {
namespace gps {

data::data(const double latitude, const double longitude, const double height,
           const double dop_precision, const gps_fix_state_t fix, const int speed_2d, const int speed_3d,
           const double ts)
    : llh_(latitude, longitude, height), dop_precision_(dop_precision),
      speed_2d_(speed_2d), speed_3d_(speed_3d), fix_(fix), ts_(ts) {
    xyz_ = openvslam::util::gps_converter::LLAToECEF(llh_);
    enu_ = openvslam::util::gps_converter::ToENU(xyz_, llh_enu_reference_);
    scaled_xyz_ = xyz_;
}

data::data(const Vec3_t& llh,  const double dop_precision,
           const gps_fix_state_t fix, const double speed_2d, const double speed_3d,
           const double ts)
    : llh_(llh), dop_precision_(dop_precision),
      speed_2d_(speed_2d), speed_3d_(speed_3d), fix_(fix), ts_(ts) {
    xyz_ = openvslam::util::gps_converter::LLAToECEF(llh_);
    enu_ = openvslam::util::gps_converter::ToENU(xyz_, llh_enu_reference_);
    scaled_xyz_ = xyz_;
}

void data::Set_XYZ(const Vec3_t &xyz) {
    xyz_ = xyz;
    scaled_xyz_ = xyz_ ;
    llh_ = openvslam::util::gps_converter::ECEFToLLA(xyz);
    enu_ = openvslam::util::gps_converter::ToENU(xyz_, llh_enu_reference_);
}

void data::Set_ENUReferenceLLH(const Vec3_t& llh_ref) {
    llh_enu_reference_ = llh_ref;
    enu_ = openvslam::util::gps_converter::ToENU(xyz_, llh_enu_reference_);
}

} // namespace gps
} // namespace openvslam
