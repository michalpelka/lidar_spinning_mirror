#include "utils.h"

Eigen::Affine3d orthogonize(const Eigen::Affine3d& p ){
    Eigen::Affine3d pose_orto(Eigen::Affine3d::Identity());
    Eigen::Quaterniond q1(p.matrix().block<3,3>(0,0)); q1.normalize();
    pose_orto.translate(p.matrix().block<3,1>(0,3));
    pose_orto.rotate(q1);
    return pose_orto;
}

Eigen::Affine3d catoptric_livox::getSE3Mirror(double angle){
    Eigen::Affine3d mirror_tf1 =Eigen::Affine3d::Identity();
    Eigen::Affine3d mirror_tf2 =Eigen::Affine3d::Identity();
    mirror_tf1.rotate(Eigen::AngleAxisd(angle-5.53747, Eigen::Vector3d::UnitX()));

    mirror_tf2.translation().x()=0.15;
    mirror_tf2.rotate(Eigen::AngleAxisd((-52.5*M_PI)/180.0, Eigen::Vector3d::UnitY()));
    return mirror_tf1*mirror_tf2;
}

Sophus::SE3d  rotating_mirror::TaitBryanPoseToSE3(const TaitBryanPose& pose){

    Eigen::Affine3d m = Eigen::Affine3d::Identity();

    double sx = sin(pose.om);
    double cx = cos(pose.om);
    double sy = sin(pose.fi);
    double cy = cos(pose.fi);
    double sz = sin(pose.ka);
    double cz = cos(pose.ka);

    m(0,0) = cy * cz;
    m(1,0) = cz * sx * sy + cx * sz;
    m(2,0) = -cx * cz * sy + sx * sz;

    m(0,1) = -cy * sz;
    m(1,1) = cx * cz - sx * sy * sz;
    m(2,1) = cz * sx + cx * sy * sz;

    m(0,2) = sy;
    m(1,2) = -cy * sx;
    m(2,2) = cx * cy;

    m(0,3) = pose.px;
    m(1,3) = pose.py;
    m(2,3) = pose.pz;

    return Sophus::SE3d(m.matrix());
}
