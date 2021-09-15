//
// Created by michal on 13.09.2021.
//

#ifndef MANDEYE_LS_COST_FUN_H
#define MANDEYE_LS_COST_FUN_H
#include <Eigen/Dense>
#include <sophus/se3.hpp>
#include <ceres/ceres.h>
#include "utils.h"

class LocalParameterizationSE3 : public ceres::LocalParameterization {
// adopted from https://github.com/strasdat/Sophus/blob/master/test/ceres/local_parameterization_se3.hpp
public:
    virtual ~LocalParameterizationSE3() {}

    // SE3 plus operation for Ceres
    //
    //  T * exp(x)
    //
    virtual bool Plus(double const* T_raw, double const* delta_raw,
                      double* T_plus_delta_raw) const {
        Eigen::Map<Sophus::SE3d const> const T(T_raw);
        Eigen::Map<Sophus::Vector6d const> const delta(delta_raw);
        Eigen::Map<Sophus::SE3d> T_plus_delta(T_plus_delta_raw);
        T_plus_delta = T * Sophus::SE3d::exp(delta);
        return true;
    }

    // Jacobian of SE3 plus operation for Ceres
    //
    // Dx T * exp(x)  with  x=0
    //
    virtual bool ComputeJacobian(double const* T_raw,
                                 double* jacobian_raw) const {
        Eigen::Map<Sophus::SE3d const> T(T_raw);
        Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> jacobian(
                jacobian_raw);
        jacobian = T.Dx_this_mul_exp_x_at_0();
        return true;
    }

    virtual int GlobalSize() const { return Sophus::SE3d::num_parameters; }

    virtual int LocalSize() const { return Sophus::SE3d::DoF; }
};

class LocalParameterizationPlane : public ceres::LocalParameterization {
public:
    virtual ~LocalParameterizationPlane() {}

    bool Plus(const double* x,
              const double* delta,
              double* x_plus_delta) const {
        x_plus_delta[0] = x[0] + delta[0];
        x_plus_delta[1] = x[1] + delta[1];
        x_plus_delta[2] = x[2] + delta[2];
        x_plus_delta[3] = x[3] + delta[3];
        Eigen::Map<Eigen::Matrix<double, 3, 1>> x_plus_deltap (x_plus_delta);
        x_plus_deltap = x_plus_deltap / x_plus_deltap.norm();
        return true;
    }
    virtual bool ComputeJacobian(double const* T_raw,
                                 double* jacobian_raw) const {
        ceres::MatrixRef(jacobian_raw, 4, 4) = ceres::Matrix::Identity(4, 4);
        return true;
    }

    virtual int GlobalSize() const { return 4; }

    virtual int LocalSize() const { return 4; }
};


template<typename T>
Eigen::Matrix<T,3,1> getPointOfRay(const Eigen::Matrix<T,4,1>& plane, const Sophus::SE3<T> &instrument_pose, const Sophus::SE3<T> &mirror_cal, const Eigen::Vector3d& ray, double angle_rad){
    const T length = T(ray.norm());
    Eigen::Affine3d mirror_tf =Eigen::Affine3d::Identity();
    mirror_tf.rotate(Eigen::AngleAxisd(angle_rad, Eigen::Vector3d::UnitX()));
    Sophus::SE3<T> m_rot_mirror = mirror_cal * Sophus::SE3<T>(mirror_tf.matrix().cast<T>());
    Eigen::Matrix<T,3,1>  rayt = ray.cast<T>() / T(length);
    Eigen::Matrix<T, 4, 1> plane_t = catoptric_livox::transformPlaneBySE3(plane,m_rot_mirror.matrix());
    Eigen::Matrix<T, 3, 1> g = instrument_pose* catoptric_livox::getMirroredRay<T>(rayt, T(length), plane_t);
    return g;
}


struct MirrorOptimization{
    const Eigen::Vector3d ray;
    const Eigen::Vector3d point_target;
    const double angle;
    MirrorOptimization(const Eigen::Vector3d& ray, const Eigen::Vector3d& point_target, const double angle):
            ray(ray), point_target(point_target), angle(angle){}

    template <typename T>
    bool operator()(const T* const mirror_plane, const T* const instrument_pose_tan, const T* const mirror_cal_tan,
                    T* residuals) const {

        const Eigen::Matrix<T,3,1> point_target_t = point_target.cast<T>();

        Eigen::Map<Sophus::SE3<T> const>  instrument_pose(instrument_pose_tan);
        Eigen::Map<Sophus::SE3<T> const>  mirror_cal(mirror_cal_tan);
        Eigen::Map<Eigen::Matrix<T,4,1>const>mirror_planet(mirror_plane);

        const Eigen::Matrix<T,3,1> pg = getPointOfRay<T>(mirror_planet, instrument_pose, mirror_cal, ray, angle );

        residuals[0] = (pg.x()-point_target_t.x());
        residuals[1] = (pg.y()-point_target_t.y());
        residuals[2] = (pg.z()-point_target_t.z());
        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Vector3d& ray, const Eigen::Vector3d& point_target,
                                       const double angle) {
        return (new ceres::AutoDiffCostFunction<MirrorOptimization, 3, 4,
                Sophus::SE3d::num_parameters,
                Sophus::SE3d::num_parameters>(
                new MirrorOptimization(ray,point_target, angle)));
//        return (new ceres::NumericDiffCostFunction<MirrorOprimizeABCDWithPose, ceres::CENTRAL, 3, 4, 4, 6, 6>(
//                new MirrorOprimizeABCDWithPose(point_1,point_target)));

//        return (new ceres::NumericDiffCostFunction<PlaneAligmentError, ceres::CENTRAL, 4, 6, 6>(
//                new PlaneAligmentError(plane_1,plane_2,pose_1,pose_2)));
    }
};

#endif //MANDEYE_LS_COST_FUN_H
