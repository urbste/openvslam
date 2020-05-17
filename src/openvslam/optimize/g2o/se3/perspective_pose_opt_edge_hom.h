#ifndef OPENVSLAM_OPTIMIZER_G2O_SE3_PERSPECTIVE_POSE_OPT_EDGE_HOM_H
#define OPENVSLAM_OPTIMIZER_G2O_SE3_PERSPECTIVE_POSE_OPT_EDGE_HOM_H

#include "openvslam/type.h"
#include "openvslam/optimize/g2o/nullspace.h"
#include "openvslam/optimize/g2o/landmark_vertex.h"
#include "openvslam/optimize/g2o/se3/shot_vertex_hom.h"
#include "openvslam/optimize/g2o/se3/SE3hom.h"

#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_unary_edge.h>
#include "openvslam/optimize/ceres/autodiff.h"

namespace openvslam {
namespace optimize {
namespace g2o {
namespace se3 {

class mono_perspective_pose_opt_edge_hom final : public ::g2o::BaseUnaryEdge<2, Vec3_t, shot_vertex_hom> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    mono_perspective_pose_opt_edge_hom();

    bool read(std::istream& is) override;

    bool write(std::ostream& os) const override;

//    void computeError() override {
//        const auto v1 = static_cast<const shot_vertex_hom*>(_vertices.at(0));
//        Vec3_t pt3_in_cam;
//        project_hom_point_to_camera(v1->estimate(), pos_w_hom_, pt3_in_cam);
//        // reduced space
//        Eigen::Matrix<double, 3, 2> pt3_in_cam_nullspace;
//        nullS_3x2_templated<double>(pt3_in_cam, pt3_in_cam_nullspace);
//        _error = pt3_in_cam.hnormalized() - _measurement.hnormalized();
//        //std::cout<<"_error: "<<_error<<std::endl;
//    }

    //void linearizeOplus();

    bool depth_is_positive() const {
        const auto v1 = static_cast<const shot_vertex_hom*>(_vertices.at(0));
        Vec3_t pt3_in_cam;
        project_hom_point_to_camera(v1->estimate(), pos_w_hom_, pt3_in_cam);

        return 0 < pt3_in_cam(2);
    }

    template<typename T>
    inline void cross(const T x[3], const T y[3], T result[3]) const
    {
      result[0] = x[1] * y[2] - x[2] * y[1];
      result[1] = x[2] * y[0] - x[0] * y[2];
      result[2] = x[0] * y[1] - x[1] * y[0];
    }

    template<typename T>
    inline T dot(const T x[3], const T y[3]) const {
        return (x[0] * y[0] + x[1] * y[1] + x[2] * y[2]);
    }

    template<typename T>
    inline T squaredNorm(const T x[3]) const {
        return dot<T>(x, x);
    }


    /**
     * templatized function to compute the error as described in the comment above
     */
    template<typename T>
    bool operator()(const T* pose, T* error) const
    {
      // normalized approximate camera ray
      // this is the homogeneous scene point back projected to the camera using [R|t]*X_hom
      // and then spherically normalized
      T point3[3];
      T point4[4];
      for (int i = 0; i < 4; ++i)
        point4[i] = T(pos_w_hom_[i]);
      // Rodrigues' formula for the rotation
      T theta = sqrt(squaredNorm(pose));

      if (theta > T(0)) {
        T v[3];
        v[0] = pose[0] / theta;
        v[1] = pose[1] / theta;
        v[2] = pose[2] / theta;
        T cth = cos(theta);
        T sth = sin(theta);

        T vXp[3];
        cross(v, point4, vXp);
        T vDotp = dot(v, point4);
        T oneMinusCth = T(1) - cth;

        point3[0] = point4[0] * cth + vXp[0] * sth + v[0] * vDotp * oneMinusCth;
        point3[1] = point4[1] * cth + vXp[1] * sth + v[1] * vDotp * oneMinusCth;
        point3[2] = point4[2] * cth + vXp[2] * sth + v[2] * vDotp * oneMinusCth;
      } else {
        // taylor expansion for theta close to zero
        T aux[3];
        cross(pose, point4, aux);
        point3[0] = point4[0] + aux[0];
        point3[1] = point4[1] + aux[1];
        point3[2] = point4[2] + aux[2];
      }

      // translation of the camera and homogeneous component
      point3[0] += (pose[3]*point4[3]);
      point3[1] += (pose[4]*point4[3]);
      point3[2] += (pose[5]*point4[3]);


      const T pt_norm = sqrt(squaredNorm(point3));
      point3[0] /= pt_norm;
      point3[1] /= pt_norm;
      point3[2] /= pt_norm;

      const T x1 = point3[0];
      const T x2 = point3[1];
      const T x3 = point3[2];
      if (x3 > T(0)) {
          const T denom = T(1) + x3;
          const T x1x2_denom = (x1*x2) / denom;
          error[0] = (T(1) - (x1*x1) / denom) * T(measurement()(0))
                     - x1x2_denom * T(measurement()(1))
                     - x1 * T(measurement()(2));

          error[1] = -  x1x2_denom * T(measurement()(0))
                     + (T(1) - (x2*x2) / denom) * T(measurement()(1))
                     - x2 * T(measurement()(2));
      } else {
          const T denom = T(1) - x3;
          const T x1x2_denom = -(x1*x2) / denom;
          error[0] =   (T(1) - (x1*x1) / denom) * T(measurement()(0))
                     + x1x2_denom * T(measurement()(1))
                     + x1 * T(measurement()(2));

          error[1] =   x1x2_denom * T(measurement()(0))
                     + (T(1) - (x2*x2) / denom) * T(measurement()(1))
                     + x2 * T(measurement()(2));
      }
    }


    void computeError()
    {
      const shot_vertex_hom* cam = static_cast<const shot_vertex_hom*>(vertex(0));
      (*this)(cam->estimate().data(), _error.data());
    }

    void linearizeOplus()
    {

      const shot_vertex_hom* camera_pose = static_cast<const shot_vertex_hom*>(vertex(0));

      // use numeric Jacobians
      //BaseBinaryEdge<2, Vector2d, VertexCameraBAL, VertexPointBAL>::linearizeOplus();
      //return;
      typedef ceres::internal::AutoDiff<mono_perspective_pose_opt_edge_hom,
              double, shot_vertex_hom::Dimension> BalAutoDiff;

      Eigen::Matrix<double, Dimension, shot_vertex_hom::Dimension, Eigen::RowMajor> dError_dCamera;
      double *parameters[] = {
          const_cast<double*>(camera_pose->estimate().data())};
      double *jacobians[] = { dError_dCamera.data()};
      double value[Dimension];
      bool diffState = BalAutoDiff::Differentiate(*this,
                                                  parameters,
                                                  Dimension, value, jacobians);

      // copy over the Jacobians (convert row-major -> column-major)
      if (diffState) {
        _jacobianOplusXi = dError_dCamera;
      } else {
        assert(0 && "Error while differentiating");
        _jacobianOplusXi.setZero();
      }
    }


    void test_jac(Eigen::Matrix<double, 2, 6>& jac) {
//        auto vi = static_cast<shot_vertex_hom*>(_vertices.at(0));
//        Eigen::Matrix<double, 3, 4> P;
//        P.fill(0.0);
//        P.block<3,3>(0,0) = Mat33_t::Identity();

//        //Mat33_t R;
//        const Vec6_t min_vec = vi->shot_vertex_hom::estimate();

//        // project point to cam and normalize to create new observation
//        // this will create x_i^a in eq. 12.225
//        Vec3_t x_i_a;
//        project_hom_point_to_camera(min_vec, pos_w_hom_, x_i_a);

//        // now we need the nullspace and the jacobian of that point
//        nullspace32_t ns_x_i_a;
//        Mat33_t jac_x_i_a;
//        nullS_3x2_templated<double>(x_i_a, ns_x_i_a);
//        jacobian_3_vec<double>(x_i_a, jac_x_i_a);

//        Eigen::Matrix<double, 2, 4> ns_js_P = ns_x_i_a.transpose() * jac_x_i_a * P;

//        Mat33_t skew_;
//        skew_mat(x_i_a, skew_);
//        Eigen::Matrix<double, 4, 6> tmp;
//        tmp.fill(0.0);
//        tmp.block<3,3>(0,0) = -skew_;
//        tmp.block<3,3>(0,3) = Mat33_t::Identity() * pos_w_hom_(3);
//        jac = -ns_js_P * tmp;
    }

    Vec4_t pos_w_hom_;
};


} // namespace se3
} // namespace g2o
} // namespace optimize
} // namespace openvslam

#endif // OPENVSLAM_OPTIMIZER_G2O_SE3_PERSPECTIVE_POSE_OPT_EDGE_H
