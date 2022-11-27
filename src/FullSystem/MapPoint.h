#pragma once

#include<iostream>
#include "util/NumType.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/utility.h"
#include "util/FrameShell.h"

#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/passthrough.h>

namespace dso
{

struct LocalPlane;

class DenseMapping
{
public:
    DenseMapping(){};
    ~DenseMapping(){};

    void updateMap(std::vector<FrameHessian*> fhs,FrameHessian* host_);
	void refineMap(std::vector<FrameHessian *> fhs, FrameHessian* dfh);
	bool acceptPatch(std::vector<Vec3f> sparsePoints, FrameHessian *dfh, LocalPlane plane);
	void keyFrameMap(std::vector<FrameHessian *> fhs, std::vector<Vec3f> &points);
	void makeMaskDistMap(float *refmask, std::vector<std::vector<Vec4f>> &clusters,
						 float *mpc_u, float *mpc_v, float *midepth, int mcount);
	void makeK(CalibHessian *HCalib);
	void makeMap(std::vector<Vec4f> cluster, FrameHessian *fh, LocalPlane& pplane);
	bool fitPlane(std::vector<Vec4f> cluster, Vec3f &dir_vector, float &dis_plane, float &score);
	bool fitPlane(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, Vec4f &pi);

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
};

class MapPoint
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	// static values
	float color;
	float weights;

	Vec3b bgr;

	float u,v;
    float idepth;
    FrameHessian *host;
    int idxInMapPoints;

	float idepth_min;
	float idepth_max;
	MapPoint(int u_, int v_, FrameHessian* host_);
	~MapPoint();
};


}