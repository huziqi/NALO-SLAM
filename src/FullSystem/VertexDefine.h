// #pragma once

// #include <g2o/core/base_vertex.h>
// #include <g2o/core/block_solver.h>
// #include <g2o/core/optimization_algorithm_levenberg.h>
// #include <g2o/solvers/csparse/linear_solver_csparse.h>
// #include <g2o/core/robust_kernel_impl.h>
// #include <g2o/types/sba/types_six_dof_expmap.h>
// #include <g2o/core/base_multi_edge.h>

// #include <opencv2/core/core.hpp>
// #include <opencv2/imgproc/imgproc.hpp>
// #include <opencv2/highgui/highgui.hpp>
// #include <opencv2/features2d/features2d.hpp>


// #include "FullSystem/FullSystem.h"
// #include "FullSystem/utility.h"
// #include "FullSystem/ResidualProjections.h"

// #include "sophus/so3.hpp"
// #include "sophus/se3.hpp"

// using namespace Sophus;
// using namespace g2o;

// namespace dso
// {
//     inline Eigen::Vector3d project2Dto3D ( int x, int y, int d, float fx, float fy, float cx, float cy, float scale )
//     {
//         float zz = float ( d ) /scale;
//         float xx = zz* ( x-cx ) /fx;
//         float yy = zz* ( y-cy ) /fy;
//         return Eigen::Vector3d ( xx, yy, zz );
//     }

//     inline Eigen::Vector2d project3Dto2D ( float x, float y, float z, float fx, float fy, float cx, float cy )
//     {
//         float u = fx*x/z+cx;
//         float v = fy*y/z+cy;
//         return Eigen::Vector2d ( u,v );
//     }

//     // 姿态结构
//     struct GrayPose {
//         GrayPose() {}

//         /// set from given data address
//         explicit GrayPose(double *data_addr) {
//             rotation = SO3d::exp(Vector3d(data_addr[0], data_addr[1], data_addr[2]));
//             translation = Vector3d(data_addr[3], data_addr[4], data_addr[5]);
//         }

//         /// 将估计值放入内存
//         void set_to(double *data_addr) {
//             auto r = rotation.log();
//             for (int i = 0; i < 3; ++i) data_addr[i] = r[i];
//             for (int i = 0; i < 3; ++i) data_addr[i + 3] = translation[i];
//         }

//         SO3d rotation;
//         Vector3d translation = Vector3d::Zero();
//     };

//     /// 位姿的顶点，6维，前三维为t，后三维为so3,
//     class VertexPose : public g2o::BaseVertex<6, GrayPose> {
//     public:
//         EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

//         VertexPose() {}

//         virtual void setToOriginImpl() override {
//             _estimate = GrayPose();
//         }

//         virtual void oplusImpl(const double *update) override {
//             _estimate.rotation = SO3d::exp(Vector3d(update[0], update[1], update[2])) * _estimate.rotation;
//             _estimate.translation += Vector3d(update[3], update[4], update[5]);
//         }

//         virtual bool read(istream &in) {}

//         virtual bool write(ostream &out) const {}
//     };

//     class VertexIdepth : public g2o::BaseVertex<1, double> {
//     public:
//         EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

//         VertexIdepth() {}

//         virtual void setToOriginImpl() override {
//             _estimate = 0;
//         }

//         virtual void oplusImpl(const double *update) override {
//             _estimate += update[0];
//         }

//         virtual bool read(istream &in) {}

//         virtual bool write(ostream &out) const {}
//     };

//     // project a 3d point into an image plane, the error is photometric error
//     // an unary edge with one vertex SE3Expmap (the pose of camera)
//     class EdgeSE3ProjectDirect: public BaseMultiEdge<patternNum, double>
//     {
//     public:
//         EIGEN_MAKE_ALIGNED_OPERATOR_NEW

//         EdgeSE3ProjectDirect() {}

//         EdgeSE3ProjectDirect (float fx, float fy, float cx, float cy, float* image, int width, int height, float u, float v)
//             :fx_ ( fx ), fy_ ( fy ), cx_ ( cx ), cy_ ( cy ), image_ ( image ), width_(width), height_(height), u_(u), v_(v)
//         {}

//         virtual void computeError()
//         {
//             auto host   = (VertexSE3Expmap *)   _vertices[0];
//             auto target = (VertexSE3Expmap *)   _vertices[1];
//             auto idepth = (VertexIdepth *)      _vertices[2];

//             // 从像素坐标系变换到host坐标系
//             Eigen::Vector3d host_pt;
//             host_pt[0] = (u_ / fx_ - cx_ / fx_) / idepth->estimate();
//             host_pt[1] = (v_ / fy_ - cy_ / fy_) / idepth->estimate();
//             host_pt[2] = 1 / idepth->estimate();

//             //从host坐标系变换到世界坐标系
//             x_world_ = host->estimate().map(host_pt);
            
//             //从世界坐标系到target坐标系
//             Eigen::Vector3d x_local = target->estimate().inverse().map(x_world_);

//             //target坐标系到像素坐标系
//             float x = x_local[0]*fx_/x_local[2] + cx_;
//             float y = x_local[1]*fy_/x_local[2] + cy_;


//             if ( x-4<0 || ( x+4 ) >width_ || ( y-4 ) <0 || ( y+4 ) >height_ )
//             {
//                 _error (0,0) = 100.0;
//                 this->setLevel ( 1 );
//             }
//             else
//             {
//                 _error (0,0) = getPixelValue ( x,y ) - _measurement;
//             }

//             // for (int i = 0; i < patternNum; i++)
//             // {
//             //     float pattern_u = x + patternP[i][0];
//             //     float pattern_v = y + patternP[i][1];
//             //     // check x,y is in the image
//             //     if ( x-4<0 || ( x+4 ) >width_ || ( y-4 ) <0 || ( y+4 ) >height_ )
//             //     {
//             //         _error (0,i) = 100.0;
//             //         this->setLevel ( 1 );
//             //     }
//             //     else
//             //     {
//             //         _error (0,i) = getPixelValue ( x,y ) - _measurement;
//             //     }
//             // }
            
//         }

//         // plus in manifold
//         virtual void linearizeOplus( )
//         {
//             if ( level() == 1 )
//             {
//                 _jacobianOplus[0] = Eigen::Matrix<double, 1, 6>::Zero();
//                 return;
//             }

//             float drescale, u, v, new_idepth;
//             float Ku, Kv;
//             Eigen::Matrix<float,3,1> KliP;
//             if(!projectPoint(point->u, point->v, point->idepth_zero_scaled, 0, 0,HCalib,
//                     PRE_RTll_0,PRE_tTll_0, drescale, u, v, Ku, Kv, KliP, new_idepth))
//             { 
//                 _jacobianOplus[0] = Eigen::Matrix<double, 1, 6>::Zero();
//                 return;
//             }

//             auto vtx = (VertexSE3Expmap*) _vertices[0];
//             Eigen::Vector3d xyz_trans = vtx->estimate().map ( x_world_ );   // q in book

//             double x = xyz_trans[0];
//             double y = xyz_trans[1];
//             double invz = 1.0/xyz_trans[2];
//             double invz_2 = invz*invz;

//             float u = x*fx_*invz + cx_;
//             float v = y*fy_*invz + cy_;

//             // jacobian from se3 to u,v
//             // NOTE that in g2o the Lie algebra is (\omega, \epsilon), where \omega is so(3) and \epsilon the translation
//             Eigen::Matrix<double, 2, 6> jacobian_uv_ksai;

//             jacobian_uv_ksai ( 0,0 ) = new_idepth*fx_;
//             jacobian_uv_ksai ( 0,1 ) = 0;
//             jacobian_uv_ksai ( 0,2 ) = -new_idepth*u*fx_;
//             jacobian_uv_ksai ( 0,3 ) = -u*v*fx_;
//             jacobian_uv_ksai ( 0,4 ) = (1+u*u)*fx_;
//             jacobian_uv_ksai ( 0,5 ) = -v*fx_;

//             jacobian_uv_ksai ( 1,0 ) = 0;
//             jacobian_uv_ksai ( 1,1 ) = new_idepth*fy_;
//             jacobian_uv_ksai ( 1,2 ) = -new_idepth*v*fy_;
//             jacobian_uv_ksai ( 1,3 ) = -(1+v*v)*fy_;
//             jacobian_uv_ksai ( 1,4 ) = u*v*fy_;
//             jacobian_uv_ksai ( 1,5 ) = u*fy_;


//             Eigen::Matrix<double, 1, 2> jacobian_pixel_uv;

//             jacobian_pixel_uv ( 0,0 ) = ( getPixelValue ( u+1,v )-getPixelValue ( u-1,v ) ) /2;
//             jacobian_pixel_uv ( 0,1 ) = ( getPixelValue ( u,v+1 )-getPixelValue ( u,v-1 ) ) /2;

//             _jacobianOplus[0] = jacobian_pixel_uv*jacobian_uv_ksai;
//         }

//         // dummy read and write functions because we don't care...
//         virtual bool read ( std::istream& in ) {}
//         virtual bool write ( std::ostream& out ) const {}

//     protected:
//         // get a gray scale value from reference image (bilinear interpolated)
//         inline float getPixelValue ( float x, float y )
//         {
//             float* data = & image_[ (int)  y  * width_ + (int) x ];
//             float xx = x - floor ( x );
//             float yy = y - floor ( y );
//             return float (
//                     ( 1-xx ) * ( 1-yy ) * data[0] +
//                     xx* ( 1-yy ) * data[1] +
//                     ( 1-xx ) *yy*data[ width_ ] +
//                     xx*yy*data[width_+1]
//                 );
//         }
//     public:
//         Eigen::Vector3d x_world_;
//         float cx_ = 0, cy_ = 0, fx_ = 0, fy_ = 0, u_=0, v_=0;
//         int width_ = 0, height_=0;
//         float* image_=nullptr;    // reference image
//     };
    

//     class EdgeProjection :
//         public g2o::BaseBinaryEdge<2, Vector2d, VertexPose, VertexPoint> {
//     public:
//         EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

//         virtual void computeError() override {
//             auto v0 = (VertexPose *) _vertices[0];
//             auto v1 = (VertexPoint *) _vertices[1];
//             auto proj = v0->project(v1->estimate());
//             _error = proj - _measurement;
//         }

//         // use numeric derivatives
//         virtual bool read(istream &in) {}

//         virtual bool write(ostream &out) const {}

//     };

// }