/**
* This file is part of DSO.
* 
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/


#pragma once

 
#include "util/NumType.h"
#include "vector"
#include <math.h>
#include <cstdlib>
#include <deque>
#include "util/settings.h"
#include "OptimizationBackend/MatrixAccumulators.h"
#include "IOWrapper/Output3DWrapper.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace std;

namespace dso
{
struct CalibHessian;
struct FrameHessian;
struct PointFrameResidual;

class CoarseTracker {
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

	CoarseTracker(int w, int h);
	~CoarseTracker();

	bool trackNewestCoarse(
			FrameHessian* newFrameHessian,
			SE3 &lastToNew_out, AffLight &aff_g2l_out,
			int coarsestLvl, Vec5 minResForAbort,
			IOWrap::Output3DWrapper* wrap=0);

	void setCoarseTrackingRef(
			std::vector<FrameHessian*> frameHessians);

	void makeK(
			CalibHessian* HCalib);

	bool debugPrint, debugPlot;

	Mat33f K[PYR_LEVELS];
	Mat33f Ki[PYR_LEVELS];
	float fx[PYR_LEVELS];
	float fy[PYR_LEVELS];
	float fxi[PYR_LEVELS];
	float fyi[PYR_LEVELS];
	float cx[PYR_LEVELS];
	float cy[PYR_LEVELS];
	float cxi[PYR_LEVELS];
	float cyi[PYR_LEVELS];
	int w[PYR_LEVELS];
	int h[PYR_LEVELS];

	// 地面的高度
	float ground_height;
	float last_height;
	int suc_num;
	float rScale;

	void debugPlotIDepthMap(float* minID, float* maxID, std::vector<IOWrap::Output3DWrapper*> &wraps);
    void debugPlotIDepthMapFloat(std::vector<IOWrap::Output3DWrapper*> &wraps);

	FrameHessian* lastRef;
	AffLight lastRef_aff_g2l;
	FrameHessian* newFrame;
	int refFrameID;

	// act as pure ouptut
	Vec5 lastResiduals;
	Vec3 lastFlowIndicators;
	double firstCoarseRMSE;
private:


	void makeCoarseDepthL0(std::vector<FrameHessian*> frameHessians);
	float* idepth[PYR_LEVELS];
	float* weightSums[PYR_LEVELS];
	float* weightSums_bak[PYR_LEVELS];


	Vec6 calcResAndGS(int lvl, Mat88 &H_out, Vec8 &b_out, const SE3 &refToNew, AffLight aff_g2l, float cutoffTH);
	Vec6 calcRes(int lvl, const SE3 &refToNew, AffLight aff_g2l, float cutoffTH);
	void calcGSSSE(int lvl, Mat88 &H_out, Vec8 &b_out, const SE3 &refToNew, AffLight aff_g2l);
	void calcGS(int lvl, Mat88 &H_out, Vec8 &b_out, const SE3 &refToNew, AffLight aff_g2l);
	void makeMaskDistMap(float *refmask, std::vector<std::vector<Vec4f>> &clusters,
						 float *mpc_u, float *mpc_v, float *midepth, int mcount);
	void delaunay(vector<vector<Vec4f>> &clusters, vector<vector<cv::Point2f>> &vertex);
	bool fitPlane(std::vector<Vec4f> cluster, Vec3f &dir_vector, float &dis_plane, float &score);
	// pc buffers
	float* pc_u[PYR_LEVELS];				//!< 每层上的有逆深度点的坐标x
	float* pc_v[PYR_LEVELS];				//!< 每层上的有逆深度点的坐标y
	float* pc_idepth[PYR_LEVELS];			//!< 每层上点的逆深度
	float* pc_color[PYR_LEVELS];			//!< 每层上点的颜色值
	int pc_n[PYR_LEVELS];					//!< 每层上点的个数

	// warped buffers
	float* buf_warped_idepth;				//!< 投影得到的点的逆深度
	float* buf_warped_u;					//!< 投影得到的归一化坐标
	float* buf_warped_v;					//!< 同上
	float* buf_warped_dx;					//!< 投影点的图像梯度
	float* buf_warped_dy;					//!< 同上
	float* buf_warped_residual;				//!< 投影得到的残差
	float* buf_warped_weight;				//!< 投影的huber函数权重
	float* buf_warped_refColor;				//!< 投影点参考帧上的灰度值
	int buf_warped_n;						//!< 投影点的个数


    std::vector<float*> ptrToDelete;


	Accumulator9 acc;
};


class CoarseDistanceMap {
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

	CoarseDistanceMap(int w, int h);
	~CoarseDistanceMap();

	void makeDistanceMap(
			std::vector<FrameHessian*> frameHessians,
			FrameHessian* frame);

	void makeInlierVotes(
			std::vector<FrameHessian*> frameHessians);

	void makeK( CalibHessian* HCalib);


	float* fwdWarpedIDDistFinal;

	Mat33f K[PYR_LEVELS];
	Mat33f Ki[PYR_LEVELS];
	float fx[PYR_LEVELS];
	float fy[PYR_LEVELS];
	float fxi[PYR_LEVELS];
	float fyi[PYR_LEVELS];
	float cx[PYR_LEVELS];
	float cy[PYR_LEVELS];
	float cxi[PYR_LEVELS];
	float cyi[PYR_LEVELS];
	int w[PYR_LEVELS];
	int h[PYR_LEVELS];

	void addIntoDistFinal(int u, int v);


private:

	PointFrameResidual** coarseProjectionGrid;
	int* coarseProjectionGridNum;
	Eigen::Vector2i* bfsList1;
	Eigen::Vector2i* bfsList2;

	void growDistBFS(int bfsNum);
};

}

