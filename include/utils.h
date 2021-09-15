#pragma once
#include <Eigen/Eigen>
#include "sophus/se3.hpp"

struct DataStream{
    double x;
    double y;
    double z;
    double angle_rad;
    double intensity;
    double timestamp;
};

struct TaitBryanPose
{
    double px;
    double py;
    double pz;
    double om;
    double fi;
    double ka;

    TaitBryanPose(){
        px = py = pz = om = fi = ka = 0.0;
    }
};

struct Plane{
    double a;
    double b;
    double c;
    double d;
    Eigen::Affine3d m;

    float distance_to_plane(Eigen::Vector3d p){
        return a * p.x() + b * p.y() + c * p.z() + d;
    }

    void from_m_to_abcd(){
        a = m(0,2);
        b = m(1,2);
        c = m(2,2);
        d = -a * m(0,3) - b * m(1,3) - c * m(2,3);
    }
    Eigen::Matrix<double, 4, 1> getABCD() const {
        return Eigen::Matrix<double, 4, 1>{a,b,c,d};
    }
    Eigen::Matrix<double, 4, 1> getABCD() {
        return Eigen::Matrix<double, 4, 1>{a,b,c,d};
    }
};


namespace rotating_mirror{
    Sophus::SE3d  TaitBryanPoseToSE3(const TaitBryanPose& pose);

}
namespace catoptric_livox {

    Eigen::Affine3d orthogonize(const Eigen::Affine3d &p);

    template<typename T>
    Eigen::Matrix<T, 4, 1> transformPlaneBySE3(const Eigen::Matrix<T, 4, 1> &plane, const Eigen::Matrix<T, 4, 4> &SE3) {

        Eigen::Matrix<T, 4, 1> r = (SE3.inverse()).transpose() * plane;
        return r;
    }

    template<typename T>
    Eigen::Matrix<T, 4, 1> getPlaneCoefFromSE3(const Eigen::Matrix<T, 4, 4> &SE3) {
        const T a = -SE3(0, 2);
        const T b = -SE3(1, 2);
        const T c = -SE3(2, 2);
        const T d = -SE3(0, 3) * a - SE3(1, 3) * b - SE3(2, 3) * c;
        return Eigen::Matrix<T, 4, 1>{a, b, c, d};
    }

    template<typename T>
    Eigen::Matrix<T, 3, 1>
    getMirroredRayIntersection(const Eigen::Matrix<T, 3, 1> &dir, T ray_length, const Eigen::Matrix<T, 4, 1> &plane) {

        Eigen::Matrix<T, 3, 1> np{plane.x(), plane.y(), plane.z()};
        np = np / np.norm();
        Eigen::Matrix<T, 3, 1> ndir = dir / dir.norm();
        const T a = np.x() * ndir.x() + np.y() * ndir.y() + np.z() * ndir.z();
        const Eigen::Matrix<T, 3, 1> intersection = -ndir * (plane.w() / a);
        return intersection;
    }


    template<typename T>
    Eigen::Matrix<T, 3, 1>
    getMirroredRay(const Eigen::Matrix<T, 3, 1> &dir, T ray_length, const Eigen::Matrix<T, 4, 1> &plane) {

        Eigen::Matrix<T, 3, 1> np{plane.x(), plane.y(), plane.z()};
        np = np / np.norm();
        Eigen::Matrix<T, 3, 1> ndir = dir / dir.norm();

        const T a = np.x() * ndir.x() + np.y() * ndir.y() + np.z() * ndir.z();
        const Eigen::Matrix<T, 3, 1> intersection = -ndir * (plane.w() / a);
        const Eigen::Matrix<T, 3, 1> rd = ndir - T(2.0) * (ndir.dot(np)) * np;
        const T ll = ray_length - intersection.norm();
        return -intersection + rd * ll;
    }

    Eigen::Affine3d getSE3Mirror(double angle);


}