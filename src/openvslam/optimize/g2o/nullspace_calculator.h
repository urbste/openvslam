#ifndef NULLSPACECALCULATOR
#define NULLSPACECALCULATOR

#include <Eigen/StdVector>
#include <Eigen/Core>
#include <Eigen/Geometry>

/**
* compute the nullspace of a 4-vector efficiently
* without QR
*
* @param vector  Eigen::Matrix<T,4,1>
*
* @return      nullspace 4x3
*/
template <typename T>
Eigen::Matrix<T, 4, 3> nullS_3x4_templated(const Eigen::Matrix<T,4,1>& vector)
{
    const T x_n = vector(3);
    const Eigen::Matrix<T, 3, 1> x_0(vector(0),vector(1),vector(2));
    const Eigen::Matrix<T, 3, 3> I_3 = Eigen::Matrix<T, 3, 3>::Identity();

    Eigen::Matrix<T, 4, 3> nullspace;
    if (x_n > T(0))
    {
        Eigen::Matrix<T, 3, 3> res = I_3 - (x_0  * x_0.transpose()) / (T(1) + x_n);
        for (int i=0; i < 3; ++i)
        {
            nullspace(3,i) = -x_0(i);
            for (int j=0; j <3; ++j)
                nullspace(i,j) = res(i,j);
        }
    }
    else
    {
        Eigen::Matrix<T, 3, 3> res = I_3 - (x_0  * x_0.transpose()) / (T(1) - x_n);
        for (int i=0; i < 3; ++i)
        {
            nullspace(3,i) = x_0(i);
            for (int j=0; j <3; ++j)
                nullspace(i,j) = res(i,j);
        }

    }
    return nullspace;
}

#endif
