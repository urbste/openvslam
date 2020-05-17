#ifndef OPENVSLAM_OPTIMIZER_G2O_SE3_PERSPECTIVE_NULLSPACE_REPROJ_EDGE_H
#define OPENVSLAM_OPTIMIZER_G2O_SE3_PERSPECTIVE_NULLSPACE_REPROJ_EDGE_H

#include <Eigen/Geometry>

#include "openvslam/type.h"
#include "openvslam/optimize/g2o/nullspace.h"
#include "openvslam/optimize/g2o/landmark_vertex4.h"
#include "openvslam/optimize/g2o/se3/shot_vertex_hom.h"
#include "openvslam/optimize/g2o/se3/SE3hom.h"

#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_unary_edge.h>

namespace openvslam {
namespace optimize {
namespace g2o {
namespace se3 {

class mono_perspective_nullspace_reproj_edge final : public ::g2o::BaseBinaryEdge<2, Vec3_t, shot_vertex_hom, landmark_vertex4> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    mono_perspective_nullspace_reproj_edge();

    bool read(std::istream& is) override;

    bool write(std::ostream& os) const override;

    void computeError() override {
        const auto v1 = static_cast<const shot_vertex_hom*>(_vertices.at(0));
        const auto v2 = static_cast<const landmark_vertex4*>(_vertices.at(1));
        Vec3_t pt3_in_cam;
        project_hom_point_to_camera(v1->estimate(), v2->estimate(), pt3_in_cam);
        // reduced space
        Eigen::Matrix<double, 3, 2> pt3_in_cam_nullspace;
        nullS_3x2_templated<double>(pt3_in_cam, pt3_in_cam_nullspace);
        _error = pt3_in_cam_nullspace.transpose() * _measurement;
    }

    //void linearizeOplus() override;

    bool depth_is_positive() const {
        const auto v1 = static_cast<const shot_vertex_hom*>(_vertices.at(0));
        const auto v2 = static_cast<const landmark_vertex4*>(_vertices.at(1));
        Vec3_t pt3_in_cam;
        project_hom_point_to_camera(v1->estimate(), v2->estimate(), pt3_in_cam);
        return 0.0 < pt3_in_cam(2);
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
    /**
     * templatized function to compute the error as described in the comment above
     */
    template<typename T>
    bool operator()(const T* pose, const T* point4, T* error) const
    {
      // normalized approximate camera ray
      // this is the homogeneous scene point back projected to the camera using [R|t]*X_hom
      // and then spherically normalized
      T point_in_cam[3];
//      const T point_w = T(point[3]);
//      for (int i = 0; i < 3; ++i)
//        point4[i] = T(point[i]);

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

        point_in_cam[0] = point4[0] * cth + vXp[0] * sth + v[0] * vDotp * oneMinusCth;
        point_in_cam[1] = point4[1] * cth + vXp[1] * sth + v[1] * vDotp * oneMinusCth;
        point_in_cam[2] = point4[2] * cth + vXp[2] * sth + v[2] * vDotp * oneMinusCth;
      } else {
        // taylor expansion for theta close to zero
        T aux[3];
        cross(pose, point4, aux);
        point_in_cam[0] = point4[0] + aux[0];
        point_in_cam[1] = point4[1] + aux[1];
        point_in_cam[2] = point4[2] + aux[2];
      }

      // translation of the camera and homogeneous component
      point_in_cam[0] = point_in_cam[0] + pose[3]*point4[3];
      point_in_cam[1] = point_in_cam[1] + pose[4]*point4[3];
      point_in_cam[2] = point_in_cam[2] + pose[5]*point4[3];

      const T pt_norm = sqrt(squaredNorm(point_in_cam));
      point_in_cam[0] /= pt_norm;
      point_in_cam[1] /= pt_norm;
      point_in_cam[2] /= pt_norm;

      //error[0] = T(measurement()(0)) / T(measurement()(2)) - point_in_cam[0];
      //error[1] = T(measurement()(1)) / T(measurement()(2)) - point_in_cam[1];

      const T x1 = point_in_cam[0];
      const T x2 = point_in_cam[1];
      const T x3 = point_in_cam[2];
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

//    void computeError()
//    {
//        const shot_vertex_hom* cam = static_cast<const shot_vertex_hom*>(vertex(0));
//        const landmark_vertex4* point = static_cast<const landmark_vertex4*>(vertex(1));

//        (*this)(cam->estimate().data(),  point->estimate().data(), _error.data());
//    }

    void linearizeOplus()
    {
      // use numeric Jacobians
      BaseBinaryEdge<2, Vec3_t, shot_vertex_hom, landmark_vertex4>::linearizeOplus();
      return;

      const shot_vertex_hom* camera_pose = static_cast<const shot_vertex_hom*>(vertex(0));
      const landmark_vertex4* point = static_cast<const landmark_vertex4*>(vertex(1));
      typedef ceres::internal::AutoDiff<mono_perspective_nullspace_reproj_edge,
              double, 6, 4> BalAutoDiff;

      Eigen::Matrix<double, 2, 6, Eigen::RowMajor> dError_dCamera;
      Eigen::Matrix<double, 2, 4, Eigen::RowMajor> dError_dPoint;
      double *parameters[] = {
          const_cast<double*>(camera_pose->estimate().data()),
          const_cast<double*>(point->estimate().data())};
      double *jacobians[] = { dError_dCamera.data(), dError_dPoint.data()};
      double value[2];
      bool diffState = BalAutoDiff::Differentiate(*this,
                                                  parameters,
                                                  2, value, jacobians);
      //std::cout<<"dError_dCamera: "<<dError_dCamera<<std::endl;
      //std::cout<<"dError_dPoint: "<<dError_dPoint<<std::endl;
      // copy over the Jacobians (convert row-major -> column-major)
      if (diffState) {
        _jacobianOplusXi = dError_dCamera;
        _jacobianOplusXj = dError_dPoint.block<2,3>(0,0);
      } else {
        //assert(0 && "Error while differentiating");
         BaseBinaryEdge<2, Vec3_t, shot_vertex_hom, landmark_vertex4>::linearizeOplus();
         std::cout<<"Error while diff\n";
         std::cout<<"dError_dCamera: "<<dError_dCamera<<std::endl;
         std::cout<<"dError_dPoint: "<<dError_dPoint<<std::endl;
        _jacobianOplusXi.setZero();
        _jacobianOplusXj.setZero();
      }
    }

    void test_jac(Eigen::Matrix<double, 2, 9>& jac) {
//        auto pose = static_cast<shot_vertex_hom*>(_vertices.at(1));
//        auto point = static_cast<landmark_vertex4*>(_vertices.at(0));

//        const Vec6_t min_vec = pose->estimate();
//        const Vec4_t point_est = point->estimate();

//        Eigen::Matrix<double, 3, 4> P;
//        P.fill(0.0);
//        P.block<3,3>(0,0) = Mat33_t::Identity();

//        Mat33_t R;
//        rotation_from_rodr_vec(min_vec.head(3), R);

//        Mat34_t cam_pose_cw;
//        R_and_t_to_hom(R, min_vec.tail(3), cam_pose_cw);
//        // project point to cam and normalize to create new observation
//        // this will create x_i^a in eq. 12.225
//        Vec3_t x_i_a;
//        project_hom_point_to_camera(min_vec, point_est, x_i_a);

//        // now we need the nullspace and the jacobian of that point
//        nullspace32_t ns_x_i_a;
//        Mat33_t jac_x_i_a;
//        nullS_3x2_templated<double>(x_i_a, ns_x_i_a);
//        jacobian_3_vec<double>(x_i_a, jac_x_i_a);

//        Eigen::Matrix<double, 2, 4> ns_js_P = ns_x_i_a.transpose() * jac_x_i_a * P;

//        jac.block<2,3>(0,0) = ns_js_P * cam_pose_cw * point->nullSpace;
//        Mat33_t skew_;
//        skew_mat(x_i_a, skew_);
//        Eigen::Matrix<double, 4, 6> tmp;
//        tmp.fill(0.0);
//        tmp.block<3,3>(0,0) = -skew_;
//        tmp.block<3,3>(0,3) = Mat33_t::Identity() * point_est(3);
//        jac.block<2,6>(0,3) = ns_js_P * tmp;
    }
};

} // namespace se3
} // namespace g2o
} // namespace optimize
} // namespace openvslam

#endif // OPENVSLAM_OPTIMIZER_G2O_SE3_PERSPECTIVE_NULLSPACE_REPROJ_EDGE_H
