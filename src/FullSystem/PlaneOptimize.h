#pragma once

#include <cmath>
#include <algorithm>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "ceres/ceres.h"
#include "ceres/cubic_interpolation.h"
#include "ceres/rotation.h"

#include "FullSystem/FullSystem.h"
#include "FullSystem/utility.h"

#include "sophus/se3.hpp"

using namespace Sophus;

namespace dso
{


class GlobalPlaneError{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    GlobalPlaneError(const float pattern_[patternNum], const ceres::BiCubicInterpolator<ceres::Grid2D<float, 1> > interpolator_, const float calib_[4], const double hit_uv_[2], const double ref_T_[6], const double pt_plane_[2]):interpolator(interpolator_), fx(calib_[0]), fy(calib_[1]), cx(calib_[2]), cy(calib_[3]), hit_u(hit_uv_[0]), hit_v(hit_uv_[1]), pt_u(pt_plane_[0]), pt_v(pt_plane_[1])
    {
        for(int i = 0; i < patternNum;i++)
            pattern[i] = pattern_[i];
        for (int i = 0; i < 6; i++)
        {
            ref_T[i] = ref_T_[i];
        }
    }

    template <typename T>
    bool operator()(const T *const plane,
                    T * residuals) const
    {
        // 把平面坐标系中的点转成世界坐标系的点
        T a = T(1) / sqrt(T(1) - plane[1] * plane[1]);
        if(ceres::IsNaN(a))
            return false;

        // v0: 平面坐标系第0坐标轴
        // v1: 平面坐标系第1坐标轴
        // plane_origin: 平面坐标系原点
        T v0[3], v1[3], plane_origin[3];
        v0[0] = -plane[2] * a;
        v0[1] = T(0);
        v0[2] = plane[0] * a;
        
        v1[0]=-plane[0] * plane[1] * a;
        v1[1]=a * (plane[0] * plane[0] + plane[2] * plane[2]);
        v1[2] = -plane[2] * plane[1] * a;

        plane_origin[0] = -plane[3] * plane[0];
        plane_origin[1] = -plane[3] * plane[1];
        plane_origin[2] = -plane[3] * plane[2];

        T p_w[3];
        p_w[0] = pt_u * v0[0] + pt_v * v1[0] + plane_origin[0];
        p_w[1] = pt_u * v0[1] + pt_v * v1[1] + plane_origin[1];
        p_w[2] = pt_u * v0[2] + pt_v * v1[2] + plane_origin[2];

        //变换到ref坐标系
        T projection[3], tr[3];
        tr[0] = T(ref_T[3]);
        tr[1] = T(ref_T[4]);
        tr[2] = T(ref_T[5]);
        ceres::AngleAxisRotatePoint(tr, p_w, projection);

        projection[0] += T(ref_T[0]);
        projection[1] += T(ref_T[1]);
        projection[2] += T(ref_T[2]);

        // 投影到像素坐标系
        T u_pro, v_pro;
        u_pro = projection[0] * T(fx) / projection[2] + T(cx);
        v_pro = projection[1] * T(fy) / projection[2] + T(cy);

        // 插值寻找投影点的灰度值
        for (int i = 0; i < patternNum; i++)
        {
            T pattern_u, pattern_v, pixel_gray_val_out;
            double pixel_test;
            pattern_u = u_pro + T(patternP[i][0]);
            pattern_v = v_pro + T(patternP[i][1]);
            if(ceres::IsNaN(pattern_u)||ceres::IsNaN(pattern_v))
                continue;

            interpolator.Evaluate(pattern_v, pattern_u, &pixel_gray_val_out);
            // interpolator.Evaluate(hit_v, hit_u, &pixel_test);
            // printf("color residual:%f\n", pattern[i] - pixel_test);
            residuals[i] = (T(pattern[i]) - pixel_gray_val_out);

            if(!ceres::IsFinite(residuals[i]))
                residuals[i] = T(1);
        }

        return true;
    }
   

    static ceres::CostFunction *Create(const float pattern[patternNum], const ceres::BiCubicInterpolator<ceres::Grid2D<float, 1> >& interpolator, const float calib[4], const double hit_uv[2], const double ref_T[6], const double pt_plane[2]){
        return (new ceres::AutoDiffCostFunction<GlobalPlaneError, patternNum, 4>(
           new GlobalPlaneError(pattern, interpolator, calib, hit_uv, ref_T, pt_plane)));
        // return (new ceres::NumericDiffCostFunction<GlobalPlaneError, ceres::CENTRAL, patternNum, 6, 6, 1>(
                // new GlobalPlaneError(pattern, interpolator, calib, uv, hit_uv)));
    }

private:
    float fx, fy, cx, cy, u, v, idepth;
    double hit_u, hit_v, pt_u, pt_v;
    float pattern[patternNum];
    double ref_T[6];
    const ceres::BiCubicInterpolator<ceres::Grid2D<float, 1>> interpolator;
};


class GreyprojectError{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    GreyprojectError(const float pattern_[patternNum], const ceres::BiCubicInterpolator<ceres::Grid2D<float, 1> > interpolator_, const float calib_[4], const float uv_[2], const double hit_uv_[2]):interpolator(interpolator_), fx(calib_[0]), fy(calib_[1]), cx(calib_[2]), cy(calib_[3]), u(uv_[0]), v(uv_[1]), hit_u(hit_uv_[0]), hit_v(hit_uv_[1])
    {
        for(int i = 0; i < patternNum;i++)
            pattern[i] = pattern_[i];
    }

    template <typename T>
    bool operator()(const T *const host,
                    const T *const target,
                    const T *const idepth,
                    T * residuals) const
    {
        // 从像素坐标系变换到host坐标系
        T point[3], hr[3];
        point[0] = T(u / fx - cx / fx) / idepth[0];
        point[1] = T(v / fy - cy / fy) / idepth[0];
        point[2] = T(1) / idepth[0];

        //从host坐标系变换到世界坐标系
        hr[0] = host[3];
        hr[1] = host[4];
        hr[2] = host[5];

        T p[3];
        ceres::AngleAxisRotatePoint(hr, point, p);
        p[0] += host[0];
        p[1] += host[1];
        p[2] += host[2];

        //对target计算逆
        T projection[3], tr[3];
        T R[9];
        tr[0] = -target[3];
        tr[1] = -target[4];
        tr[2] = -target[5];
        ceres::AngleAxisRotatePoint(tr, p, projection);
        ceres::AngleAxisToRotationMatrix(tr, R);

        T t1 = target[0];
        T t2 = target[1];
        T t3 = target[2];

        projection[0] -= t1 * R[0] + t2 * R[3] + t3 * R[6];
        projection[1] -= t1 * R[1] + t2 * R[4] + t3 * R[7];
        projection[2] -= t1 * R[2] + t2 * R[5] + t3 * R[8];

        // 投影到像素坐标系
        T u_pro, v_pro;
        u_pro = projection[0] * T(fx) / projection[2] + T(cx);
        v_pro = projection[1] * T(fy) / projection[2] + T(cy);

        // 插值寻找投影点的灰度值
        for (int i = 0; i < patternNum; i++)
        {
            T pattern_u, pattern_v, pixel_gray_val_out;
            double pixel_test;
            pattern_u = u_pro + T(patternP[i][0]);
            pattern_v = v_pro + T(patternP[i][1]);
            if(ceres::IsNaN(pattern_u)||ceres::IsNaN(pattern_v))
                continue;

            interpolator.Evaluate(pattern_v, pattern_u, &pixel_gray_val_out);
            // interpolator.Evaluate(hit_v, hit_u, &pixel_test);
            // printf("color residual:%f\n", pattern[i] - pixel_test);
            residuals[i] = (T(pattern[i]) - pixel_gray_val_out);

            if(!ceres::IsFinite(residuals[i]))
                residuals[i] = T(1);
        }

        return true;
    }
   

    static ceres::CostFunction *Create(const float pattern[patternNum], const ceres::BiCubicInterpolator<ceres::Grid2D<float, 1> >& interpolator, const float calib[4], const float uv[2], const double hit_uv[2]){
        return (new ceres::AutoDiffCostFunction<GreyprojectError, patternNum, 6, 6, 1>(
           new GreyprojectError(pattern, interpolator, calib, uv, hit_uv)));
        // return (new ceres::NumericDiffCostFunction<GreyprojectError, ceres::CENTRAL, patternNum, 6, 6, 1>(
                // new GreyprojectError(pattern, interpolator, calib, uv, hit_uv)));
    }

private:
    float fx, fy, cx, cy, u, v;
    double hit_u, hit_v;
    float pattern[patternNum];
    const ceres::BiCubicInterpolator<ceres::Grid2D<float, 1> > interpolator;
};

class ScaleConstraintError{
public:
    ScaleConstraintError(const float uv_[2], const float pi_[4], const float calib_[4]):u(uv_[0]), v(uv_[1]), pi_1(pi_[0]), pi_2(pi_[1]), pi_3(pi_[2]), pi_4(pi_[3]), fx(calib_[0]), fy(calib_[1]), cx(calib_[2]), cy(calib_[3]){}

    template <typename T>
    bool operator()(const T *const idepth,
                    T * residuals) const
    {
        T residual;

        RespointInPlane(u, v, pi_1, pi_2, pi_3, pi_4, idepth, residual);

        residuals[0] = residual;

        return true;
    }

// plane : 4 dims array
    // [0-3] : plane parameter
    // Ppoint: point on the plane, 3 dims [u,v,idepth]
    template <typename T>
    T RespointInPlane(const float u, const float v, const float pi1, const float pi2, const float pi3, const float pi4, const T *idepth, T &residual) const
    {
        // 从像素坐标系变换到相机坐标系
        T p[3];
        p[0] = T(u / fx - cx / fx) / idepth[0];
        p[1] = T(v / fy - cy / fy) / idepth[0];
        p[2] = T(1) / idepth[0];

        residual = T(pi1) * p[0] + T(pi2) * p[1] + T(pi3) * p[2] - T(pi4);
    }

    static ceres::CostFunction *Create(const float uv_[2], const float pi_[4], const float calib_[4]){
        return (new ceres::AutoDiffCostFunction<ScaleConstraintError, 1, 1>(
            new ScaleConstraintError(uv_, pi_,calib_)));
    }

private:
    float u, v, pi_1, pi_2, pi_3, pi_4;
    float fx, fy, cx, cy;
};

class PlaneLocalParameterization : public ceres::LocalParameterization
{
    virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const;
    virtual bool ComputeJacobian(const double *x, double *jacobian) const;
    virtual int GlobalSize() const { return 4; };
    virtual int LocalSize() const { return 3; };
};

//@ 把host上的点变换到target上，并检查是否在边界内
EIGEN_STRONG_INLINE bool projectPoint(
		const float &u_pt,const float &v_pt,
		const float &idepth,
		const Mat33 &KRKi, const Vec3 &Kt,
        double& Ku, double& Kv)
{
    Vec3 ptp = KRKi * Vec3(u_pt, v_pt, 1) + Kt * idepth; // host上点除深度
    Ku = ptp[0] / ptp[2];
	Kv = ptp[1] / ptp[2];
    // printf("ku, kv:%f, %f\n", Ku, Kv);
    return Ku > 1.1f && Kv > 1.1f && Ku < wM3G && Kv < hM3G; // 不在边缘
}

EIGEN_STRONG_INLINE void projectPoint(const float &u_pt, const float &v_pt,
        const float &idepth,
        const int &dx, const int &dy,
        CalibHessian *const &HCalib,
        const Mat33 &R, const Vec3 &t,
        float &drescale, float &u, float &v,
        float &Ku, float &Kv, Vec3 &KliP, float &new_idepth)
{
    KliP = Vec3(
			(u_pt+dx-HCalib->cxl())*HCalib->fxli(),
			(v_pt+dy-HCalib->cyl())*HCalib->fyli(),
			1);

	Vec3 ptp = R * KliP + t*idepth;
	drescale = 1.0f/ptp[2];
	new_idepth = idepth*drescale;

	u = ptp[0] * drescale;
	v = ptp[1] * drescale;
	Ku = u*HCalib->fxl() + HCalib->cxl();
	Kv = v*HCalib->fyl() + HCalib->cyl();
}
class GrayTHFactor_TH : public ceres::SizedCostFunction<1, 6, 6, 1>
{
    public:
        GrayTHFactor_TH(FrameHessian* host_, FrameHessian* target_, PointHessian* point_, CalibHessian* HCalib_): 
        host(host_), target(target_), point(point_), HCalib(HCalib_){}

        virtual ~GrayTHFactor_TH() {}

        virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;


    private:
        FrameHessian *host, *target;
        PointHessian *point;
        CalibHessian *HCalib;
};
class GrayTHFactor_H2T : public ceres::SizedCostFunction<1, 6, 1>
{
    public:
        GrayTHFactor_H2T(FrameHessian* host_, FrameHessian* target_, PointHessian* point_, CalibHessian* HCalib_): 
        host(host_), target(target_), point(point_), HCalib(HCalib_){}

        virtual ~GrayTHFactor_H2T() {}

        virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;


    private:
        FrameHessian *host, *target;
        PointHessian *point;
        CalibHessian *HCalib;
};
class GrayTHFactor_T2H : public ceres::SizedCostFunction<1, 6, 1>
{
    public:
        GrayTHFactor_T2H(FrameHessian* host_, FrameHessian* target_, PointHessian* point_, CalibHessian* HCalib_): 
        host(host_), target(target_), point(point_), HCalib(HCalib_){}

        virtual ~GrayTHFactor_T2H() {}

        virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;


    private:
        FrameHessian *host, *target;
        PointHessian *point;
        CalibHessian *HCalib;
};

bool GrayTHFactor_TH::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    Eigen::Vector3d ti(parameters[0][0], parameters[0][1], parameters[0][2]);
    SO3d Ri = SO3d::exp(Vector3d(parameters[0][3], parameters[0][4], parameters[0][5]));

    Eigen::Vector3d tj(parameters[1][0], parameters[1][1], parameters[1][2]);
    SO3d Rj = SO3d::exp(Vector3d(parameters[1][3], parameters[1][4], parameters[1][5]));

    SE3 T_i, T_j, T_ij;
    T_i.setRotationMatrix(Ri.matrix());
    T_i.translation() = ti;
    T_j.setRotationMatrix(Rj.matrix());
    T_j.translation() = tj;

    T_ij = T_j * T_i.inverse();

    double inv_dep_i = parameters[2][0];

    float drescale, u, v, new_idepth;
    float Ku, Kv;
    Vec3 KliP;

    projectPoint(point->u, point->v, inv_dep_i, 0, 0, HCalib, T_ij.rotationMatrix(), T_ij.translation(), drescale, u, v, Ku, Kv, KliP, new_idepth);

    bool all_zero = 0;
    if (!(Ku > 1.1f && Kv > 1.1f && Ku < wM3G && Kv < hM3G)||inv_dep_i<0)
        all_zero = 1;

    // 计算灰度残差
    float gray = point->color[4];
    Vec3f hitColor = Vec3f::Zero();
    if (!all_zero)
    {
        Vec3f hitColor = (getInterpolatedElement33(target->dI, Ku, Kv, wG[0]));
        residuals[0] = hitColor[0] - gray;
    }
    else
    {
        residuals[0] = 100;
    }

    // 伴随矩阵
    SE3 hostToTarget = target->get_worldToCam_evalPT() * host->get_worldToCam_evalPT().inverse();
    Mat66 AH = Mat66::Identity();
    Mat66 AT = Mat66::Identity();

    AH = -hostToTarget.Adj();
    AT = Mat66::Identity();

    //! 图像导数 dx dy
    Vec6 d_xi_x, d_xi_y;
    Vec2 d_uv;
    d_uv[0] = double(hitColor[1]);
    d_uv[1] = double(hitColor[2]);

    // 像素对位姿导数
    d_xi_x[0] = new_idepth*HCalib->fxl();
    d_xi_x[1] = 0;
    d_xi_x[2] = -new_idepth*u*HCalib->fxl();
    d_xi_x[3] = -u*v*HCalib->fxl();
    d_xi_x[4] = (1+u*u)*HCalib->fxl();
    d_xi_x[5] = -v*HCalib->fxl();

    d_xi_y[0] = 0;
    d_xi_y[1] = new_idepth*HCalib->fyl();
    d_xi_y[2] = -new_idepth*v*HCalib->fyl();
    d_xi_y[3] = -(1+v*v)*HCalib->fyl();
    d_xi_y[4] = u*v*HCalib->fyl();
    d_xi_y[5] = u*HCalib->fyl();

    // 像素对逆深度导数
    float d_d_x = drescale * (T_ij.translation()[0]-T_ij.translation()[2]*u)*HCalib->fxl();
    float d_d_y = drescale * (T_ij.translation()[1]-T_ij.translation()[2]*v)*HCalib->fyl();

    if(!jacobians)
        return true;
    
    if(jacobians[0])
    {
        Eigen::Map<Eigen::Matrix<double, 1, 6>> jacobian_pose_i(jacobians[0]);

        if(!all_zero)
        {
            jacobian_pose_i = d_uv[0] * d_xi_x.transpose() + d_uv[1] * d_xi_y.transpose();
            jacobian_pose_i = jacobian_pose_i * AH;
        }
        else
            jacobian_pose_i.setZero();
    }
    if(jacobians[1])
    {
        Eigen::Map<Eigen::Matrix<double, 1, 6>> jacobian_pose_j(jacobians[1]);

        if(!all_zero)
        {
            jacobian_pose_j = d_uv[0] * d_xi_x.transpose() + d_uv[1] * d_xi_y.transpose();
            jacobian_pose_j = jacobian_pose_j * AT;
        }
        else
            jacobian_pose_j.setZero();
    }
    if(jacobians[2])
    {
        double *jacobian_idepth = jacobians[2];
        
        if(!all_zero)
            jacobian_idepth[0] = d_uv[0] * d_d_x + d_uv[1] * d_d_y;
        else
            jacobian_idepth[0] = 0;
    }

    return true;
}


bool GrayTHFactor_H2T::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    Eigen::Vector3d ti(parameters[0][0], parameters[0][1], parameters[0][2]);
    SO3d Ri = SO3d::exp(Vector3d(parameters[0][3], parameters[0][4], parameters[0][5]));

    Eigen::Vector3d tj(parameters[1][0], parameters[1][1], parameters[1][2]);
    SO3d Rj = SO3d::exp(Vector3d(parameters[1][3], parameters[1][4], parameters[1][5]));

    SE3 T_i, T_j, T_ij;
    T_i.setRotationMatrix(Ri.matrix());
    T_i.translation() = ti;
    T_j.setRotationMatrix(Rj.matrix());
    T_j.translation() = tj;

    T_ij = T_j.inverse() * T_i;

    double inv_dep_i = parameters[3][0];
    double depth = 1.0 / inv_dep_i;//host帧下的深度

    float drescale, u, v, new_idepth;
    float Ku, Kv;
    Vec3 KliP;

    projectPoint(point->u, point->v, point->idepth_zero_scaled, 0, 0, HCalib, T_ij.rotationMatrix(), T_ij.translation(), drescale, u, v, Ku, Kv, KliP, new_idepth);

    // 计算灰度残差
    float gray = point->color[4];
    Vec3f hitColor = (getInterpolatedElement33(target->dI, Ku, Kv, wG[0]));
    residuals[0] = hitColor[0] - gray;

    // 伴随矩阵
    SE3 hostToTarget = target->get_worldToCam_evalPT() * host->get_worldToCam_evalPT().inverse();
    Mat66 AH = Mat66::Identity();
    Mat66 AT = Mat66::Identity();

    AH = -hostToTarget.Adj();
    AT = Mat66::Identity();

    //! 图像导数 dx dy
    Vec6 d_xi_x, d_xi_y;
    Vec2 d_uv;
    d_uv[0] = double(hitColor[1]);
    d_uv[1] = double(hitColor[2]);

    // 像素对位姿导数
    d_xi_x[0] = new_idepth*HCalib->fxl();
    d_xi_x[1] = 0;
    d_xi_x[2] = -new_idepth*u*HCalib->fxl();
    d_xi_x[3] = -u*v*HCalib->fxl();
    d_xi_x[4] = (1+u*u)*HCalib->fxl();
    d_xi_x[5] = -v*HCalib->fxl();

    d_xi_y[0] = 0;
    d_xi_y[1] = new_idepth*HCalib->fyl();
    d_xi_y[2] = -new_idepth*v*HCalib->fyl();
    d_xi_y[3] = -(1+v*v)*HCalib->fyl();
    d_xi_y[4] = u*v*HCalib->fyl();
    d_xi_y[5] = u*HCalib->fyl();

    // 像素对逆深度导数
    float d_d_x = drescale * (T_ij.translation()[0]-T_ij.translation()[2]*u)*HCalib->fxl();
    float d_d_y = drescale * (T_ij.translation()[1]-T_ij.translation()[2]*v)*HCalib->fyl();

    if(!jacobians)
        return true;
    
    if(jacobians[0])
    {
        Eigen::Map<Eigen::Matrix<double, 1, 6>> jacobian_pose_i(jacobians[0]);
        jacobian_pose_i = d_uv[0] * d_xi_x.transpose() + d_uv[1] * d_xi_y.transpose();
        jacobian_pose_i = jacobian_pose_i * AH;
    }
    if(jacobians[1])
    {
        Eigen::Map<Eigen::Matrix<double, 1, 6>> jacobian_pose_j(jacobians[1]);
        jacobian_pose_j = d_uv[0] * d_xi_x.transpose() + d_uv[1] * d_xi_y.transpose();
        jacobian_pose_j = jacobian_pose_j * AT;
    }
    if(jacobians[2])
    {
        double *jacobian_idepth = jacobians[2];
        jacobian_idepth[0] = d_uv[0] * d_d_x + d_uv[1] * d_d_y;
    }

    return true;
}
}