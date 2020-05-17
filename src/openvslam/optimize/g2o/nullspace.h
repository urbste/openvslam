#ifndef OPENVSLAM_OPTIMIZER_G2O_NULLSPACE_H
#define OPENVSLAM_OPTIMIZER_G2O_NULLSPACE_H

#include <Eigen/StdVector>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <iostream>
#include "openvslam/type.h"

using namespace std;

namespace openvslam {
namespace optimize {
namespace g2o {

// see Förstner et al, PCV, page 521
// Js(x)
template <typename T>
void jacobian_3_vec(
        const Eigen::Matrix<T, 3, 1>& vec,
        Eigen::Matrix<T, 3, 3>& jac) {
    jac = (Eigen::Matrix<T, 3, 3>::Identity() -
           (vec * vec.transpose()) / (vec.transpose() * vec)) / vec.norm();
}


// have a look at
// W. Foerstner, PCV, Page 370
// and S.Urban, MLPnP paper
template <typename T>
void get_information_for_bearing(const T& variance,
                                 const Eigen::Matrix<T, 3, 3>& inv_cam_mat,
                                 const Eigen::Matrix<T, 3, 3>& bearing_jac,
                                 const Eigen::Matrix<T, 3, 2>& bearing_ns,
                                 Eigen::Matrix<T, 2, 2>& information)
{
    const Eigen::Matrix<T, 3, 3> Exx = Eigen::Matrix<T,3,1>(variance, variance, 0).asDiagonal();
    const Eigen::Matrix<T, 3, 3> proj_Exx = inv_cam_mat * Exx * inv_cam_mat.transpose();
    const Eigen::Matrix<T, 3, 3> Evv = bearing_jac * proj_Exx * bearing_jac.transpose();
    const Eigen::Matrix<T, 2, 2> Ers = bearing_ns.transpose() * Evv * bearing_ns;
    information = Ers.inverse();
}

/**
* compute the nullspace of a 3-vector efficiently
* without QR see W.Foerstner PCV, Page 778, eq. A.120)
*
* @param vector  Eigen::Matrix<T, 3, 1>
*
* @return      nullspace 3x2
*/
template <typename T>
void nullS_3x2_templated(const Eigen::Matrix<T, 3, 1>& vector,
                         Eigen::Matrix<T, 3, 2>& nullspace)
{
    const T x_n = vector(2);
    const Eigen::Matrix<T, 2, 1> x_0(vector(0),vector(1));
    const Eigen::Matrix<T, 2, 2> I_2 = Eigen::Matrix<T, 2, 2>::Identity();

    if (x_n > T(0))
    {
        const Eigen::Matrix<T, 2, 2> tmp =
                (I_2 - (x_0  * x_0.transpose()) / (T(1) + x_n));
        nullspace.row(0) = tmp.row(0);
        nullspace.row(1) = tmp.row(1);
        nullspace.row(2) = -x_0.transpose();
    }
    else
    {
        const Eigen::Matrix<T, 2, 2> tmp =
                (I_2 - (x_0  * x_0.transpose()) / (T(1) - x_n));
        nullspace.row(0) = tmp.row(0);
        nullspace.row(1) = tmp.row(1);
        nullspace.row(2) = x_0.transpose();
    }
}

/**
* compute the nullspace of a 4-vector efficiently
* without QR, see W.Foerstner PCV, Page 778, eq. A.120)
*
* @param vector  Eigen::Matrix<T, 4, 1>
*
* @return      nullspace 4x3
*/
template <typename T>
void nullS_3x4_templated(const Eigen::Matrix<T, 4, 1>& vector,
                         Eigen::Matrix<T, 4, 3>& nullspace)
{
    const T x_n = vector(3);
    const Eigen::Matrix<T, 3, 1> x_0(vector(0),vector(1),vector(2));
    const Eigen::Matrix<T, 3, 3> I_3 = Eigen::Matrix<T, 3, 3>::Identity();

    if (x_n > T(0))
    {
        const Eigen::Matrix<T, 3, 3> tmp =
                (I_3 - (x_0  * x_0.transpose()) / (T(1) + x_n));
        nullspace.row(0) = tmp.row(0);
        nullspace.row(1) = tmp.row(1);
        nullspace.row(2) = tmp.row(2);
        nullspace.row(3) = -x_0.transpose();
    }
    else
    {
        const Eigen::Matrix<T, 3, 3> tmp =
                (I_3 - (x_0  * x_0.transpose()) / (T(1) - x_n));
        nullspace.row(0) = tmp.row(0);
        nullspace.row(1) = tmp.row(1);
        nullspace.row(2) = tmp.row(2);
        nullspace.row(3) = x_0.transpose();
    }
}
}
}
}
#endif // OPENVSLAM_OPTIMIZER_G2O_NULLSPACE_H
