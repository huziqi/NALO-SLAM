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


/*
 * KFBuffer.cpp
 *
 *  Created on: Jan 7, 2014
 *      Author: engelj
 */

#include "FullSystem/CoarseTracker.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/Residuals.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"
#include "IOWrapper/ImageRW.h"
#include <algorithm>

#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>

using namespace std;
using namespace cv;

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso
{


template<int b, typename T>
T* allocAligned(int size, std::vector<T*> &rawPtrVec)
{
    const int padT = 1 + ((1 << b)/sizeof(T));
    T* ptr = new T[size + padT];
    rawPtrVec.push_back(ptr);
    T* alignedPtr = (T*)(( ((uintptr_t)(ptr+padT)) >> b) << b);
    return alignedPtr;
}


CoarseTracker::CoarseTracker(int ww, int hh) : lastRef_aff_g2l(0,0)
{
	// make coarse tracking templates.
	for(int lvl=0; lvl<pyrLevelsUsed; lvl++)
	{
		int wl = ww>>lvl;
        int hl = hh>>lvl;

        idepth[lvl] = allocAligned<4,float>(wl*hl, ptrToDelete);
        weightSums[lvl] = allocAligned<4,float>(wl*hl, ptrToDelete);
        weightSums_bak[lvl] = allocAligned<4,float>(wl*hl, ptrToDelete);

        pc_u[lvl] = allocAligned<4,float>(wl*hl, ptrToDelete);
        pc_v[lvl] = allocAligned<4,float>(wl*hl, ptrToDelete);
        pc_idepth[lvl] = allocAligned<4,float>(wl*hl, ptrToDelete);
        pc_color[lvl] = allocAligned<4,float>(wl*hl, ptrToDelete);

	}

	// warped buffers
    buf_warped_idepth = allocAligned<4,float>(ww*hh, ptrToDelete);
    buf_warped_u = allocAligned<4,float>(ww*hh, ptrToDelete);
    buf_warped_v = allocAligned<4,float>(ww*hh, ptrToDelete);
    buf_warped_dx = allocAligned<4,float>(ww*hh, ptrToDelete);
    buf_warped_dy = allocAligned<4,float>(ww*hh, ptrToDelete);
    buf_warped_residual = allocAligned<4,float>(ww*hh, ptrToDelete);
    buf_warped_weight = allocAligned<4,float>(ww*hh, ptrToDelete);
    buf_warped_refColor = allocAligned<4,float>(ww*hh, ptrToDelete);


	newFrame = 0;
	lastRef = 0;
	debugPlot = debugPrint = true;
	w[0]=h[0]=0;
	refFrameID=-1;
	ground_height = -1;
	last_height = -1;
	suc_num = 0;
	rScale = 1;
}
CoarseTracker::~CoarseTracker()
{
    for(float* ptr : ptrToDelete)
        delete[] ptr;
    ptrToDelete.clear();
}

void CoarseTracker::makeK(CalibHessian* HCalib)
{
	w[0] = wG[0];
	h[0] = hG[0];

	fx[0] = HCalib->fxl();
	fy[0] = HCalib->fyl();
	cx[0] = HCalib->cxl();
	cy[0] = HCalib->cyl();

	for (int level = 1; level < pyrLevelsUsed; ++ level)
	{
		w[level] = w[0] >> level;
		h[level] = h[0] >> level;
		fx[level] = fx[level-1] * 0.5;
		fy[level] = fy[level-1] * 0.5;
		cx[level] = (cx[0] + 0.5) / ((int)1<<level) - 0.5;
		cy[level] = (cy[0] + 0.5) / ((int)1<<level) - 0.5;
	}

	for (int level = 0; level < pyrLevelsUsed; ++ level)
	{
		K[level]  << fx[level], 0.0, cx[level], 0.0, fy[level], cy[level], 0.0, 0.0, 1.0;
		Ki[level] = K[level].inverse();
		fxi[level] = Ki[level](0,0);
		fyi[level] = Ki[level](1,1);
		cxi[level] = Ki[level](0,2);
		cyi[level] = Ki[level](1,2);
	}
}

/**
 * @brief 根据像素点集三角剖分
 * 
 * @param clusters：投影点x坐标，投影点y坐标，投影点逆深度，投影点在mask上的像素值
 * @param vertex: 三角剖分出来的网格
*/
void CoarseTracker::delaunay(std::vector<std::vector<Vec4f>>& clusters, vector<vector<cv::Point2f>>& vertex)
{
	for (int i = 0; i < clusters.size();i++)
	{
		Vec4f temp;
		temp = *min_element(clusters[i].begin(), clusters[i].end(), [](Vec4f a, Vec4f b){ return a[0] < b[0]; });
		int xmin = temp[0];
		temp = *max_element(clusters[i].begin(), clusters[i].end(), [](Vec4f a, Vec4f b){ return a[0] > b[0]; });
		int xmax = temp[0];
		temp = *min_element(clusters[i].begin(), clusters[i].end(), [](Vec4f a, Vec4f b){ return a[1] < b[1]; });
		int ymin = temp[1];
		temp = *max_element(clusters[i].begin(), clusters[i].end(), [](Vec4f a, Vec4f b){ return a[1] > b[1]; });
		int ymax = temp[1];

		Rect rect(xmin, ymin, xmax-xmin, ymax-ymin);
		Subdiv2D subdiv(rect);
		

		for (auto pp : clusters[i])
		{
			Point2f fp(pp[0], pp[1]);
			
		}
	}
}


/**
 * @brief 在mask上找到像素值一样的点，统计点的个数，并排序
 * 
 * @param clusters：投影点x坐标，投影点y坐标，投影点逆深度，投影点在mask上的像素值
*/
void CoarseTracker::makeMaskDistMap(float* refmask, std::vector<std::vector<Vec4f>>& clusters, 
									float* mpc_u, float* mpc_v, float* midepth, int mcount)
{
	int width = w[0];
	int height = h[0];

	std::vector<Vec4f> waitVec;
	std::vector<Vec4f> _waitVec;
	std::vector<Vec4f> readyVec;
	waitVec.reserve(mcount);
	_waitVec.reserve(mcount);
	readyVec.reserve(mcount);

	// interpolation
	for (int i = 0; i < mcount;i++)
	{
		int xx, yy;
		Vec4f point;

		int ix = (int)mpc_u[i];
		int iy = (int)mpc_v[i];
		float dx = mpc_u[i] - ix;
		float dy = mpc_v[i] - iy;

		if(dx<0.5) xx = ix;
		else xx = ix++;
		if(dy<0.5) yy = iy;
		else yy = iy++;

		point << xx, yy, midepth[i], refmask[xx+yy*w[0]];
		//printf("%f, ",refmask[xx+yy*w[0]]);

		if(xx>2 && xx<width-2 && yy>2 && yy<height-2)
			waitVec.push_back(point);
	}

	// for(int i=0;i<w[0];i++)
	// {
	// 	for (int j = 0; j < h[0];j++)
	// 	{
	// 		float c = refmask[i+j*width];
	// 		//if(c==155.0)
	// 		printf("%f, ",c);
	// 	}
	// }
	
	//显示mask
	if(maskplot)
	{
		MinimalImageB3 mas(w[0],h[0]);

		for(int i=0;i<w[0]*h[0];i++)
		{
			float c = refmask[i];
			if (c > 255)
				c = 255;
			mas.at(i) = Vec3b(c,c,c);
		}
		IOWrap::displayImage("Mask", &mas);
	}
	

	//printf("waitvec的大小：%ld\n", waitVec.size());

	while (!waitVec.empty())
	{
		readyVec.clear();
		_waitVec.clear();

		readyVec.push_back(waitVec.back());
		waitVec.pop_back();

		for (int idx = waitVec.size() - 1; idx >= 0; idx--)
		{
			if (readyVec.back()[3]==waitVec[idx][3])
			{
				readyVec.push_back(waitVec[idx]);
				waitVec.pop_back();
			}
			else
			{
				_waitVec.push_back(waitVec[idx]);
				waitVec.pop_back();
			}
		}
		waitVec.swap(_waitVec);
		clusters.push_back(readyVec);
	}
	// debug
	// printf("clusters size:%ld\n", clusters.size());
	// for (int i = 0; i < clusters.size();i++)
	// {
	// 	printf("cluster %d size: %ld, value: %f\n", i, clusters[i].size(),clusters[i][0][3]);
	// }

	// 根据相同像素值的点的数量逆序排序
	sort(clusters.begin(), clusters.end(), [&](std::vector<Vec4f> i, std::vector<Vec4f> j){ return i.size() > j.size(); });

	// for (int i = 0; i < clusters.size();i++)
	// {
	// 	printf("cluster的数量：%d\n", clusters[i].size());
	// }
	
	
	return;
}

/**
 * @brief 根据点的逆深度，拟合点云的平面参数
 * 
 * @param clusters：投影点x坐标，投影点y坐标，投影点逆深度，投影点在mask上的像素值
 * @param dir_vector: 方向向量
 * @param dis_plane: 平面距相机坐标系原点距离
*/
bool CoarseTracker::fitPlane(std::vector<Vec4f> cluster, Vec3f& dir_vector, float& dis_plane, float &score)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

	Vec3f x_axis, z_axis;
	// float mid_x = 0;
	float mid_z = 0;
	float dot_product = 0;
	size_t size_points = cluster.size();

	x_axis << 1, 0, 0;
	z_axis << 0, 0, 1;

	
	for(auto pt:cluster)
	{
		pcl::PointXYZ point;

		//将像素投影到相机坐标系
		Vec3f mP = Ki[0] * Vec3f(pt[0], pt[1], 1);
		mP /= pt[2];
		if(!std::isfinite(mP[0]) || !std::isfinite(mP[1]) || !std::isfinite(mP[2]))
		{
			printf("fitplane投影到相机坐标系,坐标值不是有限值!\n");
			return 0;
		}

		point.x = mP[0];
		point.y = mP[1];
		point.z = mP[2];
		cloud->points.push_back(point);

		// mid_x += point.x / size_points;
		mid_z += point.z / size_points;
	}

	//printf("cloud size:%ld\n", cloud->points.size());
	if(cloud->points.size()<20)//kitti_code
	// if(cloud->points.size()<5)
		return false;

	pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
	pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
	// Create the segmentation object
	pcl::SACSegmentation<pcl::PointXYZ> seg;
	// Optional
	seg.setOptimizeCoefficients (true);
	// Mandatory
	seg.setModelType (pcl::SACMODEL_PLANE);//kitti_code
	seg.setMethodType (pcl::SAC_RANSAC);//kitti_code
	seg.setDistanceThreshold (0.01);//kitti_code
	// seg.setAxis(Eigen::Vector3f(0, -1, 0));
	// seg.setDistanceThreshold(0.05);

	seg.setInputCloud (cloud);
	seg.segment (*inliers, *coefficients);

	dir_vector[0] = coefficients->values[0];
	dir_vector[1] = coefficients->values[1];
	dir_vector[2] = coefficients->values[2];
	dis_plane = coefficients->values[3];
	// printf("plane para:%lf,%lf,%lf,%lf\n", dir_vector[0], dir_vector[1], dir_vector[2], dis_plane);

	// 计算地面得分
	if(cluster.size()<100 || mid_z<0 || cluster[0][3]<200)//kitti_code
	// if(cluster.size()<50 || mid_z<0)
	{
		score = 9999999;
		return true;
	}

	dot_product = x_axis.transpose() * dir_vector;
	dot_product += z_axis.transpose() * dir_vector;

	score = dot_product * 1000 + abs(dis_plane) * 100 + 100 / cluster.size(); // kitti_code
	// score = dot_product * 1000 - dis_plane * 100;


	return true;
}

// #######[!!!!!!! 深度图传递 !!!!!!!!!!!!!!!]
//@ 使用在当前帧上投影的点的逆深度, 来生成每个金字塔层上点的逆深度值
void CoarseTracker::makeCoarseDepthL0(std::vector<FrameHessian*> frameHessians)
{
	// make coarse tracking templates for latstRef.
	memset(idepth[0], 0, sizeof(float)*w[0]*h[0]);
	memset(weightSums[0], 0, sizeof(float)*w[0]*h[0]);
//[ ***step 1*** ] 计算其它点在最新帧投影第0层上的各个像素的逆深度权重, 和加权逆深度
	for(FrameHessian* fh : frameHessians)
	{
		for(PointHessian* ph : fh->pointHessians)
		{
			if(ph->lastResiduals[0].first != 0 && ph->lastResiduals[0].second == ResState::IN)
			{
				PointFrameResidual* r = ph->lastResiduals[0].first;
				assert(r->efResidual->isActive() && r->target == lastRef);// target是最新的关键帧
				int u = r->centerProjectedTo[0] + 0.5f; // 四舍五入
				int v = r->centerProjectedTo[1] + 0.5f;
				float new_idepth = r->centerProjectedTo[2];
				float weight = sqrtf(1e-3 / (ph->efPoint->HdiF+1e-12)); // 协方差逆做权重

				idepth[0][u+w[0]*v] += new_idepth *weight;
				weightSums[0][u+w[0]*v] += weight;
			}
		}
	}

//[ ***step 2*** ] 从下层向上层生成逆深度和权重
	for(int lvl=1; lvl<pyrLevelsUsed; lvl++)
	{
		int lvlm1 = lvl-1;
		int wl = w[lvl], hl = h[lvl], wlm1 = w[lvlm1];

		float* idepth_l = idepth[lvl];
		float* weightSums_l = weightSums[lvl];

		float* idepth_lm = idepth[lvlm1];
		float* weightSums_lm = weightSums[lvlm1];

		for(int y=0;y<hl;y++)
			for(int x=0;x<wl;x++)
			{
				int bidx = 2*x   + 2*y*wlm1;
				idepth_l[x + y*wl] = 		idepth_lm[bidx] +
											idepth_lm[bidx+1] +
											idepth_lm[bidx+wlm1] +
											idepth_lm[bidx+wlm1+1];

				weightSums_l[x + y*wl] = 	weightSums_lm[bidx] +
											weightSums_lm[bidx+1] +
											weightSums_lm[bidx+wlm1] +
											weightSums_lm[bidx+wlm1+1];
			}
	}

//[ ***step 3*** ] 0和1层 对于没有深度的像素点, 使用周围斜45度的四个点来填充
    // dilate idepth by 1.
	for(int lvl=0; lvl<2; lvl++)
	{
		int numIts = 1;


		for(int it=0;it<numIts;it++)
		{
			int wh = w[lvl]*h[lvl]-w[lvl];
			int wl = w[lvl];
			float* weightSumsl = weightSums[lvl];
			float* weightSumsl_bak = weightSums_bak[lvl];
			memcpy(weightSumsl_bak, weightSumsl, w[lvl]*h[lvl]*sizeof(float));
			float* idepthl = idepth[lvl];	// dotnt need to make a temp copy of depth, since I only
											// read values with weightSumsl>0, and write ones with weightSumsl<=0.
			for(int i=w[lvl];i<wh;i++)
			{
				if(weightSumsl_bak[i] <= 0)
				{
					float sum=0, num=0, numn=0;
					if(weightSumsl_bak[i+1+wl] > 0) { sum += idepthl[i+1+wl]; num+=weightSumsl_bak[i+1+wl]; numn++;}
					if(weightSumsl_bak[i-1-wl] > 0) { sum += idepthl[i-1-wl]; num+=weightSumsl_bak[i-1-wl]; numn++;}
					if(weightSumsl_bak[i+wl-1] > 0) { sum += idepthl[i+wl-1]; num+=weightSumsl_bak[i+wl-1]; numn++;}
					if(weightSumsl_bak[i-wl+1] > 0) { sum += idepthl[i-wl+1]; num+=weightSumsl_bak[i-wl+1]; numn++;}
					if(numn>0) {idepthl[i] = sum/numn; weightSumsl[i] = num/numn;}
				}
			}
		}
	}

//[ ***step 4*** ] 2层向上, 对于没有深度的像素点, 使用上下左右的四个点来填充
	// dilate idepth by 1 (2 on lower levels).
	for(int lvl=2; lvl<pyrLevelsUsed; lvl++)
	{
		int wh = w[lvl]*h[lvl]-w[lvl];
		int wl = w[lvl];
		float* weightSumsl = weightSums[lvl];
		float* weightSumsl_bak = weightSums_bak[lvl];
		memcpy(weightSumsl_bak, weightSumsl, w[lvl]*h[lvl]*sizeof(float));
		float* idepthl = idepth[lvl];	// dotnt need to make a temp copy of depth, since I only
										// read values with weightSumsl>0, and write ones with weightSumsl<=0.
		for(int i=w[lvl];i<wh;i++)
		{
			if(weightSumsl_bak[i] <= 0)
			{
				float sum=0, num=0, numn=0;
				if(weightSumsl_bak[i+1] > 0) { sum += idepthl[i+1]; num+=weightSumsl_bak[i+1]; numn++;}
				if(weightSumsl_bak[i-1] > 0) { sum += idepthl[i-1]; num+=weightSumsl_bak[i-1]; numn++;}
				if(weightSumsl_bak[i+wl] > 0) { sum += idepthl[i+wl]; num+=weightSumsl_bak[i+wl]; numn++;}
				if(weightSumsl_bak[i-wl] > 0) { sum += idepthl[i-wl]; num+=weightSumsl_bak[i-wl]; numn++;}
				if(numn>0) {idepthl[i] = sum/numn; weightSumsl[i] = num/numn;}
			}
		}
	}

//[ ***step 5*** ] 归一化点的逆深度并赋值给成员变量pc_*
	// normalize idepths and weights.
	for(int lvl=0; lvl<pyrLevelsUsed; lvl++)
	{
		float* weightSumsl = weightSums[lvl];
		float* idepthl = idepth[lvl];
		Eigen::Vector3f* dIRefl = lastRef->dIp[lvl];

		int wl = w[lvl], hl = h[lvl];

		int lpc_n=0;
		float* lpc_u = pc_u[lvl];
		float* lpc_v = pc_v[lvl];
		float* lpc_idepth = pc_idepth[lvl];
		float* lpc_color = pc_color[lvl];


		for(int y=2;y<hl-2;y++)
			for(int x=2;x<wl-2;x++)
			{
				int i = x+y*wl;

				// 逆深度不为零的点
				if(weightSumsl[i] > 0)
				{
					idepthl[i] /= weightSumsl[i];
					lpc_u[lpc_n] = x;
					lpc_v[lpc_n] = y;
					lpc_idepth[lpc_n] = idepthl[i];
					lpc_color[lpc_n] = dIRefl[i][0];



					if(!std::isfinite(lpc_color[lpc_n]) || !(idepthl[i]>0))
					{
						idepthl[i] = -1;
						continue;	// just skip if something is wrong.
					}
					lpc_n++;
				}
				else
					idepthl[i] = -1;

				weightSumsl[i] = 1;
			}

		pc_n[lvl] = lpc_n;
	}

//[ ***step 6*** ] 只对第一层恢复更多的跟踪点，try！！
	if(dense_track)
	{
		MinimalImageB3 img_out(w[0],h[0]);
		for(int i=0;i<w[0]*h[0];i++)
		{
			float c = frameHessians.back()->dI[i][0]*0.7;
			if(c>255) c=255;
			img_out.at(i) = Vec3b(c,c,c);
		}
		int ori_num = pc_n[0];

		std::vector<std::vector<Vec4f>> clusters;
		float *mpc_u = pc_u[0];
		float *mpc_v = pc_v[0];
		float *midepth = pc_idepth[0];
		
		// 找到mask中权重值相同的点，并排序
		//printf("一开始pc_n的数量为：%d, frame id: %d\n",pc_n[0], frameHessians.back()->frameID);
		makeMaskDistMap(frameHessians.back()->mask, clusters, mpc_u, mpc_v, midepth, pc_n[0]);

		// delaunay(clusters, vertex);

		if(clusters.size()<4)
		{
			//printf("clusters的数量太少了:%ld！！！！\n",clusters.size());
			return;
		}

		float* mlpc_u = pc_u[0];
		float* mlpc_v = pc_v[0];
		float* mlpc_idepth = pc_idepth[0];
		float* mlpc_color = pc_color[0];
		Eigen::Vector3f* mdIRefl = lastRef->dIp[0];
		float min_score = FLT_MAX;

		// 地面参数（相对于相机坐标系）[pi1, pi2, pi3, pi4]
		Vec4f gp_raw = Vec4f::Zero();
		std::vector<int> g_u, g_v;
		g_u.reserve(w[0] * h[0]/2);
		g_v.reserve(w[0] * h[0]/2);

		for (int i = 0; i < clusters.size();i++)//4只是测试值，试验！！！，后期需要根据cluster实际的大小动态调整
		{
			Vec3f dir_vector;
			float dis_plane;
			float score;

			dir_vector << 1, 1, 1;
			dis_plane = 1;

			if(!fitPlane(clusters[i], dir_vector, dis_plane, score))
				continue;

			// printf("plane%d [%lf, %lf, %lf]:平面得分：%f,平面高度:%f\n",i, dir_vector[0],dir_vector[1],dir_vector[2], score, dis_plane);

			// 找到点集中x,y的最大最小值
			int minx = 9999;
			int miny = 9999;
			int maxx = -1;
			int maxy = -1;
			for (auto clu : clusters[i])
			{
				if(clu[0]>maxx) maxx = clu[0];
				if(clu[0]<minx) minx = clu[0];
				if(clu[1]>maxy) maxy = clu[1];
				if(clu[1]<miny) miny = clu[1];
			}

			if(score<min_score)
			{
				if(dir_vector[1]>0)
					gp_raw << -dir_vector[0], -dir_vector[1], -dir_vector[2], -dis_plane;
				else
					gp_raw << dir_vector[0], dir_vector[1], dir_vector[2], dis_plane;
				ground_height = abs(dis_plane);
				min_score = score;

				// 更新g_u,g_v,g_rect
				g_u.clear(); g_v.clear();
				for (auto cc : clusters[i])
				{
					g_u.push_back((int)cc[0]);
					g_v.push_back((int)cc[1]);
				}
			}

			if(maxx>w[0]-1||minx<1||maxy>h[0]-1||miny<1)
				continue;

			// 在固定范围内随机恢复平面上的点，试验！！！，后期可以根据梯度选点！！！
			int last_pcn = pc_n[0];
			int refMaskColor = clusters[i][0][3];
			//printf("refMaskcolor: %d, direct vector:(%f,%f,%f)\n", refMaskColor,dir_vector[0],dir_vector[1],dir_vector[2]);
			
			if (refMaskColor==0)
				continue;
			for (int x = minx; x < maxx; x++)
				for (int y = miny; y < maxy;y++)
				{
					if(frameHessians.back()->mask[x+y*w[0]]!=refMaskColor)
					{
						//printf("continue!!!!!!!!\n");
						continue;
					}
					
					if(x%5==0&&y%5==0)//均匀选点
					{
						float new_idepth = dir_vector.transpose() * Ki[0] * Vec3f(x, y, 1);
						new_idepth /= -dis_plane;

						mlpc_u[pc_n[0]+1] = x;
						mlpc_v[pc_n[0]+1] = y;
						mlpc_idepth[pc_n[0]+1] = new_idepth;
						mlpc_color[pc_n[0]+1] = mdIRefl[x+y*w[0]][0];
						pc_n[0]++;
					}
				}

			for (int tt = last_pcn; tt < pc_n[0];tt++)
			{
				if(pc_u[0][tt]<2||pc_u[0][tt]>w[0]-2||pc_v[0][tt]<2||pc_v[0][tt]>h[0]-2)
					continue;
				img_out.setPixelCirc(pc_u[0][tt], pc_v[0][tt], Vec3b(0, 255, 0));
			}
					
		}

		// printf("ground raw--->[%lf, %lf, %lf, %lf]\n", gp_raw[0], gp_raw[1], gp_raw[2], gp_raw[3]);
		// 统计onground的点的数量, 并标记pointhessian中的地面点
		int temp_gp_num = 0;
		for (FrameHessian *fh : frameHessians)
		{
			for(PointHessian* ph : fh->pointHessians)
			{
				if(ph->lastResiduals[0].first != 0 && ph->lastResiduals[0].second == ResState::IN)
				{
					PointFrameResidual* r = ph->lastResiduals[0].first;
					assert(r->efResidual->isActive() && r->target == lastRef);// target是最新的关键帧
					int u = r->centerProjectedTo[0] + 0.5f; // 四舍五入
					int v = r->centerProjectedTo[1] + 0.5f;

					for (int gi = 0; gi < g_u.size(); gi++)
					{
						if(u==g_u[gi]&&v==g_v[gi])
						{
							ph->onground = true;
							temp_gp_num++;
							break;
						}
					}
				}
			}
		}
		//printf("地面上的点一共有:%d\n", temp_gp_num);

		if (!scale_fix)
		{
			if (last_height < 0)
				last_height = ground_height;
			else
			{
				float diff = abs(last_height - ground_height);
				if (diff < 0.01)//kitti_code
				// if(diff < 0.1)
					suc_num++;
				else
					suc_num = 0;
			}
			if (suc_num > 3) //连续三个地面高度变化小于0.01，则将高度固定
			{
				init_height = (ground_height + last_height) / 2;
				scale_fix = true;
				printf("ground height fix at:%f!!!!!!!!!!\n", init_height);
			}
			last_height = ground_height;
		}
		else
		{
			// 去除尺度比例中的外点
			float scale_rate = ground_height / init_height;
			if (last_ScaleRate < 0)
			{
				last_ScaleRate = scale_rate;
				last_gp[0] = gp_raw[0];
				last_gp[1] = gp_raw[1];
				last_gp[2] = gp_raw[2];
				last_gp[3] = gp_raw[3];
				old_rate.push_back(last_ScaleRate);
			}
			else
			{
				float ave2, ave3, ave4, ave5;
				ave2 = ave3 = ave4 = ave5 = scale_rate;

				int size = old_rate.size();
				if (size > 5)
				{
					ave2 = abs(old_rate[size - 1] + old_rate[size - 2]) / 2;
					ave3 = abs(old_rate[size - 1] + old_rate[size - 2] + old_rate[size - 3]) / 3;
					ave4 = abs(old_rate[size - 1] + old_rate[size - 2] + old_rate[size - 3] + old_rate[size - 4]) / 4;
					ave5 = abs(old_rate[size - 1] + old_rate[size - 2] + old_rate[size - 3] + old_rate[size - 4] + old_rate[size - 5]) / 5;
				}
				else if (size > 4)
				{
					ave2 = abs(old_rate[size - 1] + old_rate[size - 2]) / 2;
					ave3 = abs(old_rate[size - 1] + old_rate[size - 2] + old_rate[size - 3]) / 3;
					ave4 = abs(old_rate[size - 1] + old_rate[size - 2] + old_rate[size - 3] + old_rate[size - 4]) / 4;
				}
				else if (size > 3)
				{
					ave2 = abs(old_rate[size - 1] + old_rate[size - 2]) / 2;
					ave3 = abs(old_rate[size - 1] + old_rate[size - 2] + old_rate[size - 3]) / 3;
				}
				else if (size > 2)
					ave2 = abs(old_rate[size - 1] + old_rate[size - 2]) / 2;

				float diff = abs(last_ScaleRate - scale_rate) / last_ScaleRate;
				float diff2 = abs(ave2 - scale_rate) / ave2;
				float diff3 = abs(ave3 - scale_rate) / ave3;
				float diff4 = abs(ave4 - scale_rate) / ave4;
				float diff5 = abs(ave5 - scale_rate) / ave5;
				if (diff > 0.25 && diff2 > 0.25 && diff3 > 0.25 && diff4 > 0.25 && diff5 > 0.25)//kitti_code
				// if (diff > 0.5 && diff2 > 0.5 && diff3 > 0.5 && diff4 > 0.5 && diff5 > 0.5)
				{
					// printf("diff too big, change scale rate %f -> %f\n", scale_rate, last_ScaleRate);
					scale_rate = last_ScaleRate;//kitti_code
					frameHessians.back()->groundP = Vec4f(last_gp[0], last_gp[1], last_gp[2], last_gp[3]);//kitti_code
					// frameHessians.back()->haveground = true;
				}
				else
				{
					// printf("@@@@@@@@@@@@@@@@@@@@@@@@@@@scale rate:%f\n", scale_rate);
					last_ScaleRate = scale_rate;
					frameHessians.back()->haveground = true;
					frameHessians.back()->groundP = gp_raw;
				}
				// rScale = 1 / scale_rate;
				// printf("scale_rate:%f, rScale:%f\n", scale_rate, rScale);

				if (old_rate.size() < 7)
				{
					old_rate.push_back(last_ScaleRate);
				}
				else
				{
					old_rate.pop_front();
					old_rate.push_back(last_ScaleRate);
				}
			}
		}
		
		if(scale_fix&&frameHessians.size()>6)//kitti_code
		{
			assert(std::isfinite(frameHessians[frameHessians.size() - 2]->groundP[0]) &&
				   std::isfinite(frameHessians[frameHessians.size() - 2]->groundP[1]) &&
				   std::isfinite(frameHessians[frameHessians.size() - 2]->groundP[2]) &&
				   std::isfinite(frameHessians[frameHessians.size() - 2]->groundP[3]));

			for (int fhid = frameHessians.size() - 2; fhid > 0;fhid--)
			{
				if ((frameHessians[fhid]->groundP[0] == 0 &&
					 frameHessians[fhid]->groundP[1] == 0 &&
					 frameHessians[fhid]->groundP[2] == 0 &&
					 frameHessians[fhid]->groundP[3] == 0)
					 ||(frameHessians[fhid]->groundP[3]==0)) continue;
				else
				{
					frameHessians.back()->last_ground[0] = frameHessians[fhid]->groundP[0];
					frameHessians.back()->last_ground[1] = frameHessians[fhid]->groundP[1];
					frameHessians.back()->last_ground[2] = frameHessians[fhid]->groundP[2];
					frameHessians.back()->last_ground[3] = frameHessians[fhid]->groundP[3];
					break;
				}
			}
			//assert(frameHessians.back()->last_ground[3] != 0);
		}


		// if(maskplot)
		// IOWrap::displayImage("expand Pixels", &img_out);
		// char buf[1000];
		// snprintf(buf, 1000, "/home/hzq/dso/image_plot/pixel_selection_%05d.png", frameHessians.back()->frameID);
		// IOWrap::writeImage(buf,&img_out);
	}
}

//@ 对跟踪的最新帧和参考帧之间的残差, 求 Hessian 和 b
void CoarseTracker::calcGSSSE(int lvl, Mat88 &H_out, Vec8 &b_out, const SE3 &refToNew, AffLight aff_g2l)
{
	acc.initialize();

	__m128 fxl = _mm_set1_ps(fx[lvl]);
	__m128 fyl = _mm_set1_ps(fy[lvl]);
	__m128 b0 = _mm_set1_ps(lastRef_aff_g2l.b);
	__m128 a = _mm_set1_ps((float)(AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure, lastRef_aff_g2l, aff_g2l)[0]));

	__m128 one = _mm_set1_ps(1);
	__m128 minusOne = _mm_set1_ps(-1);
	__m128 zero = _mm_set1_ps(0);

	int n = buf_warped_n;
	assert(n%4==0);
	for(int i=0;i<n;i+=4)
	{
		__m128 dx = _mm_mul_ps(_mm_load_ps(buf_warped_dx+i), fxl);
		__m128 dy = _mm_mul_ps(_mm_load_ps(buf_warped_dy+i), fyl);
		__m128 u = _mm_load_ps(buf_warped_u+i);
		__m128 v = _mm_load_ps(buf_warped_v+i);
		__m128 id = _mm_load_ps(buf_warped_idepth+i);


		acc.updateSSE_eighted(
				_mm_mul_ps(id,dx),
				_mm_mul_ps(id,dy),
				_mm_sub_ps(zero, _mm_mul_ps(id,_mm_add_ps(_mm_mul_ps(u,dx), _mm_mul_ps(v,dy)))),
				_mm_sub_ps(zero, _mm_add_ps(
						_mm_mul_ps(_mm_mul_ps(u,v),dx),
						_mm_mul_ps(dy,_mm_add_ps(one, _mm_mul_ps(v,v))))),
				_mm_add_ps(
						_mm_mul_ps(_mm_mul_ps(u,v),dy),
						_mm_mul_ps(dx,_mm_add_ps(one, _mm_mul_ps(u,u)))),
				_mm_sub_ps(_mm_mul_ps(u,dy), _mm_mul_ps(v,dx)),
				_mm_mul_ps(a,_mm_sub_ps(b0, _mm_load_ps(buf_warped_refColor+i))),
				minusOne,
				_mm_load_ps(buf_warped_residual+i),
				_mm_load_ps(buf_warped_weight+i));
	}

	acc.finish();
	H_out = acc.H.topLeftCorner<8,8>().cast<double>() * (1.0f/n);
	b_out = acc.H.topRightCorner<8,1>().cast<double>() * (1.0f/n);

	H_out.block<8,3>(0,0) *= SCALE_XI_ROT;
	H_out.block<8,3>(0,3) *= SCALE_XI_TRANS;
	H_out.block<8,1>(0,6) *= SCALE_A;
	H_out.block<8,1>(0,7) *= SCALE_B;
	H_out.block<3,8>(0,0) *= SCALE_XI_ROT;
	H_out.block<3,8>(3,0) *= SCALE_XI_TRANS;
	H_out.block<1,8>(6,0) *= SCALE_A;
	H_out.block<1,8>(7,0) *= SCALE_B;
	b_out.segment<3>(0) *= SCALE_XI_ROT;
	b_out.segment<3>(3) *= SCALE_XI_TRANS;
	b_out.segment<1>(6) *= SCALE_A;
	b_out.segment<1>(7) *= SCALE_B;
}


// #########[!!!!!!!!!! 计算残差 !!!!!!!!!!]########
//@ 计算当前位姿投影得到的残差(能量值), 并进行一些统计
//! 构造尽量多的点, 有助于跟踪
Vec6 CoarseTracker::calcRes(int lvl, const SE3 &refToNew, AffLight aff_g2l, float cutoffTH)
{
	float E = 0;
	int numTermsInE = 0;
	int numTermsInWarped = 0;
	int numSaturated=0;

	int wl = w[lvl];
	int hl = h[lvl];
	Eigen::Vector3f* dINewl = newFrame->dIp[lvl];
	float fxl = fx[lvl];
	float fyl = fy[lvl];
	float cxl = cx[lvl];
	float cyl = cy[lvl];


	Mat33f RKi = (refToNew.rotationMatrix().cast<float>() * Ki[lvl]);
	Vec3f t = (refToNew.translation()).cast<float>();
	Vec2f affLL = AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure, lastRef_aff_g2l, aff_g2l).cast<float>();


	float sumSquaredShiftT=0;
	float sumSquaredShiftRT=0;
	float sumSquaredShiftNum=0;

	float maxEnergy = 2*setting_huberTH*cutoffTH-setting_huberTH*setting_huberTH;	// energy for r=setting_coarseCutoffTH.


    MinimalImageB3* resImage = 0;
	if(debugPlot)
	{
		resImage = new MinimalImageB3(wl,hl);
		resImage->setConst(Vec3b(255,255,255));
	}

	//* 投影在ref帧上的点
	int nl = pc_n[lvl];
	float* lpc_u = pc_u[lvl];
	float* lpc_v = pc_v[lvl];
	float* lpc_idepth = pc_idepth[lvl];
	float* lpc_color = pc_color[lvl];

	// printf("一共这么多投影点：%d\n", nl);
	for (int i = 0; i < nl; i++)
	{
		float id = lpc_idepth[i];
		float x = lpc_u[i];
		float y = lpc_v[i];

		// 参考帧投影到当前帧
		Vec3f pt = RKi * Vec3f(x, y, 1) + t*id;
		float u = pt[0] / pt[2];
		float v = pt[1] / pt[2];
		float Ku = fxl * u + cxl;
		float Kv = fyl * v + cyl;
		float new_idepth = id/pt[2];

		if(lvl==0 && i%32==0)
		{
			// translation only (positive)
			Vec3f ptT = Ki[lvl] * Vec3f(x, y, 1) + t*id;
			float uT = ptT[0] / ptT[2];
			float vT = ptT[1] / ptT[2];
			float KuT = fxl * uT + cxl;
			float KvT = fyl * vT + cyl;

			// translation only (negative)
			Vec3f ptT2 = Ki[lvl] * Vec3f(x, y, 1) - t*id;
			float uT2 = ptT2[0] / ptT2[2];
			float vT2 = ptT2[1] / ptT2[2];
			float KuT2 = fxl * uT2 + cxl;
			float KvT2 = fyl * vT2 + cyl;

			//translation and rotation (negative)
			Vec3f pt3 = RKi * Vec3f(x, y, 1) - t*id;
			float u3 = pt3[0] / pt3[2];
			float v3 = pt3[1] / pt3[2];
			float Ku3 = fxl * u3 + cxl;
			float Kv3 = fyl * v3 + cyl;

			//translation and rotation (positive)
			//already have it.

			sumSquaredShiftT += (KuT-x)*(KuT-x) + (KvT-y)*(KvT-y);
			sumSquaredShiftT += (KuT2-x)*(KuT2-x) + (KvT2-y)*(KvT2-y);
			sumSquaredShiftRT += (Ku-x)*(Ku-x) + (Kv-y)*(Kv-y);
			sumSquaredShiftRT += (Ku3-x)*(Ku3-x) + (Kv3-y)*(Kv3-y);
			sumSquaredShiftNum+=2;
		}

		if(!(Ku > 2 && Kv > 2 && Ku < wl-3 && Kv < hl-3 && new_idepth > 0)) continue;



		float refColor = lpc_color[i];
        Vec3f hitColor = getInterpolatedElement33(dINewl, Ku, Kv, wl); // 新帧上插值
        if(!std::isfinite((float)hitColor[0])) continue;
        float residual = hitColor[0] - (float)(affLL[0] * refColor + affLL[1]);
        float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);


		if(fabs(residual) > cutoffTH)
		{
			if(debugPlot) resImage->setPixel4(lpc_u[i], lpc_v[i], Vec3b(0,0,255));
			E += maxEnergy;
			numTermsInE++;
			numSaturated++;
		}
		else
		{
			if(debugPlot) resImage->setPixel4(lpc_u[i], lpc_v[i], Vec3b(residual+128,residual+128,residual+128));

			E += hw *residual*residual*(2-hw);
			numTermsInE++;

			buf_warped_idepth[numTermsInWarped] = new_idepth;
			buf_warped_u[numTermsInWarped] = u;
			buf_warped_v[numTermsInWarped] = v;
			buf_warped_dx[numTermsInWarped] = hitColor[1];
			buf_warped_dy[numTermsInWarped] = hitColor[2];
			buf_warped_residual[numTermsInWarped] = residual;
			buf_warped_weight[numTermsInWarped] = hw;
			buf_warped_refColor[numTermsInWarped] = lpc_color[i];
			numTermsInWarped++;
		}
	}

	while(numTermsInWarped%4!=0)
	{
		buf_warped_idepth[numTermsInWarped] = 0;
		buf_warped_u[numTermsInWarped] = 0;
		buf_warped_v[numTermsInWarped] = 0;
		buf_warped_dx[numTermsInWarped] = 0;
		buf_warped_dy[numTermsInWarped] = 0;
		buf_warped_residual[numTermsInWarped] = 0;
		buf_warped_weight[numTermsInWarped] = 0;
		buf_warped_refColor[numTermsInWarped] = 0;
		numTermsInWarped++;
	}
	buf_warped_n = numTermsInWarped;


	if(debugPlot)
	{
		IOWrap::displayImage("RES", resImage, false);
		IOWrap::waitKey(0);
		delete resImage;
	}

	Vec6 rs;
	rs[0] = E;												// 投影的能量值
	rs[1] = numTermsInE;									// 投影的点的数目
	rs[2] = sumSquaredShiftT/(sumSquaredShiftNum+0.1);		// 纯平移时 平均像素移动的大小
	rs[3] = 0;
	rs[4] = sumSquaredShiftRT/(sumSquaredShiftNum+0.1);		// 平移+旋转 平均像素移动大小
	rs[5] = numSaturated / (float)numTermsInE;   			// 大于cutoff阈值的百分比

	return rs;
}



void CoarseTracker::setCoarseTrackingRef(
		std::vector<FrameHessian*> frameHessians)
{
	assert(frameHessians.size()>0);
	lastRef = frameHessians.back();
	makeCoarseDepthL0(frameHessians);// 生成逆深度估值



	refFrameID = lastRef->shell->id;
	lastRef_aff_g2l = lastRef->aff_g2l();

	firstCoarseRMSE=-1;

}



// ########[!!!!跟踪最新帧!!!!!!]##########
//@ 对新来的帧进行跟踪, 优化得到位姿, 光度参数
bool CoarseTracker::trackNewestCoarse(
		FrameHessian* newFrameHessian,
		SE3 &lastToNew_out, AffLight &aff_g2l_out,
		int coarsestLvl,
		Vec5 minResForAbort,
		IOWrap::Output3DWrapper* wrap)
{
	debugPlot = setting_render_displayCoarseTrackingFull;
	debugPrint = false;

	assert(coarsestLvl < 5 && coarsestLvl < pyrLevelsUsed);

	lastResiduals.setConstant(NAN);
	lastFlowIndicators.setConstant(1000);


	newFrame = newFrameHessian;
	int maxIterations[] = {10,20,50,50,50};// 不同层迭代的次数
	float lambdaExtrapolationLimit = 0.001;

	SE3 refToNew_current = lastToNew_out;// 优化的初始值
	AffLight aff_g2l_current = aff_g2l_out;

	bool haveRepeated = false;

	//* 使用金字塔进行跟踪, 从顶层向下开始跟踪
	for(int lvl=coarsestLvl; lvl>=0; lvl--)
	{
		Mat88 H; Vec8 b;
		float levelCutoffRepeat=1;
		//[ ***step 1*** ] 计算残差, 保证最多60%残差大于阈值, 计算正规方程
		Vec6 resOld = calcRes(lvl, refToNew_current, aff_g2l_current, setting_coarseCutoffTH*levelCutoffRepeat);
		//* 保证大于阈值的点小于60%
		while(resOld[5] > 0.6 && levelCutoffRepeat < 50)
		{
			levelCutoffRepeat*=2;// 超过阈值的多, 则放大阈值重新计算
			resOld = calcRes(lvl, refToNew_current, aff_g2l_current, setting_coarseCutoffTH*levelCutoffRepeat);

            if(!setting_debugout_runquiet)
                printf("INCREASING cutoff to %f (ratio is %f)!\n", setting_coarseCutoffTH*levelCutoffRepeat, resOld[5]);
		}

		calcGSSSE(lvl, H, b, refToNew_current, aff_g2l_current);

		float lambda = 0.01;

		if(debugPrint)
		{
			Vec2f relAff = AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure, lastRef_aff_g2l, aff_g2l_current).cast<float>();
			printf("lvl%d, it %d (l=%f / %f) %s: %.3f->%.3f (%d -> %d) (|inc| = %f)! \t",
					lvl, -1, lambda, 1.0f,
					"INITIA",
					0.0f,
					resOld[0] / resOld[1],
					 0,(int)resOld[1],
					0.0f);
			std::cout << refToNew_current.log().transpose() << " AFF " << aff_g2l_current.vec().transpose() <<" (rel " << relAff.transpose() << ")\n";
		}

		//[ ***step 2*** ] 迭代优化
		for(int iteration=0; iteration < maxIterations[lvl]; iteration++)
		{
			//[ ***step 2.1*** ] 计算增量
			Mat88 Hl = H;
			for(int i=0;i<8;i++) Hl(i,i) *= (1+lambda);
			Vec8 inc = Hl.ldlt().solve(-b);

			if(setting_affineOptModeA < 0 && setting_affineOptModeB < 0)	// fix a, b
			{
				inc.head<6>() = Hl.topLeftCorner<6,6>().ldlt().solve(-b.head<6>());
			 	inc.tail<2>().setZero();
			}
			if(!(setting_affineOptModeA < 0) && setting_affineOptModeB < 0)	// fix b
			{
				inc.head<7>() = Hl.topLeftCorner<7,7>().ldlt().solve(-b.head<7>());
			 	inc.tail<1>().setZero();
			}
			if(setting_affineOptModeA < 0 && !(setting_affineOptModeB < 0))	// fix a
			{
				Mat88 HlStitch = Hl;
				Vec8 bStitch = b;
				HlStitch.col(6) = HlStitch.col(7);
				HlStitch.row(6) = HlStitch.row(7);
				bStitch[6] = bStitch[7];
				Vec7 incStitch = HlStitch.topLeftCorner<7,7>().ldlt().solve(-bStitch.head<7>());
				inc.setZero();
				inc.head<6>() = incStitch.head<6>();
				inc[6] = 0;
				inc[7] = incStitch[6];
			}




			float extrapFac = 1;
			if(lambda < lambdaExtrapolationLimit) extrapFac = sqrt(sqrt(lambdaExtrapolationLimit / lambda));
			inc *= extrapFac;

			Vec8 incScaled = inc;
			incScaled.segment<3>(0) *= SCALE_XI_ROT;
			incScaled.segment<3>(3) *= SCALE_XI_TRANS;
			incScaled.segment<1>(6) *= SCALE_A;
			incScaled.segment<1>(7) *= SCALE_B;

            if(!std::isfinite(incScaled.sum())) incScaled.setZero();

			SE3 refToNew_new = SE3::exp((Vec6)(incScaled.head<6>())) * refToNew_current;
			AffLight aff_g2l_new = aff_g2l_current;
			aff_g2l_new.a += incScaled[6];
			aff_g2l_new.b += incScaled[7];

			Vec6 resNew = calcRes(lvl, refToNew_new, aff_g2l_new, setting_coarseCutoffTH*levelCutoffRepeat);

			bool accept = (resNew[0] / resNew[1]) < (resOld[0] / resOld[1]);

			if(debugPrint)
			{
				Vec2f relAff = AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure, lastRef_aff_g2l, aff_g2l_new).cast<float>();
				printf("lvl %d, it %d (l=%f / %f) %s: %.3f->%.3f (%d -> %d) (|inc| = %f)! \t",
						lvl, iteration, lambda,
						extrapFac,
						(accept ? "ACCEPT" : "REJECT"),
						resOld[0] / resOld[1],
						resNew[0] / resNew[1],
						(int)resOld[1], (int)resNew[1],
						inc.norm());
				std::cout << refToNew_new.log().transpose() << " AFF " << aff_g2l_new.vec().transpose() <<" (rel " << relAff.transpose() << ")\n";
			}
			//[ ***step 2.3*** ] 接受则求正规方程, 继续迭代, 优化到增量足够小
			if(accept)
			{
				calcGSSSE(lvl, H, b, refToNew_new, aff_g2l_new);
				resOld = resNew;
				aff_g2l_current = aff_g2l_new;
				refToNew_current = refToNew_new;
				lambda *= 0.5;
			}
			else
			{
				lambda *= 4;
				if(lambda < lambdaExtrapolationLimit) lambda = lambdaExtrapolationLimit;
			}

			if(!(inc.norm() > 1e-3))
			{
				if(debugPrint)
					printf("inc too small, break!\n");
				break;
			}
		}
		//[ ***step 3*** ] 记录上一次残差, 光流指示, 如果调整过阈值则重新计算这一层
		// set last residual for that level, as well as flow indicators.
		lastResiduals[lvl] = sqrtf((float)(resOld[0] / resOld[1]));
		lastFlowIndicators = resOld.segment<3>(2);
		if(lastResiduals[lvl] > 1.5*minResForAbort[lvl]) return false;


		if(levelCutoffRepeat > 1 && !haveRepeated)
		{
			lvl++;// 这一层重新算一遍
			haveRepeated=true;
			//printf("REPEAT LEVEL!\n");
		}
	}

	// set!
	lastToNew_out = refToNew_current;
	aff_g2l_out = aff_g2l_current;

	//[ ***step 4*** ] 判断优化失败情况
	if((setting_affineOptModeA != 0 && (fabsf(aff_g2l_out.a) > 1.2))
	|| (setting_affineOptModeB != 0 && (fabsf(aff_g2l_out.b) > 200)))
		return false;

	Vec2f relAff = AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure, lastRef_aff_g2l, aff_g2l_out).cast<float>();

	if((setting_affineOptModeA == 0 && (fabsf(logf((float)relAff[0])) > 1.5))
	|| (setting_affineOptModeB == 0 && (fabsf((float)relAff[1]) > 200)))
		return false;



	if(setting_affineOptModeA < 0) aff_g2l_out.a=0;
	if(setting_affineOptModeB < 0) aff_g2l_out.b=0;

	return true;
}



void CoarseTracker::debugPlotIDepthMap(float* minID_pt, float* maxID_pt, std::vector<IOWrap::Output3DWrapper*> &wraps)
{
    if(w[1] == 0) return;


	int lvl = 0;

	{
		std::vector<float> allID;
		for(int i=0;i<h[lvl]*w[lvl];i++)
		{
			if(idepth[lvl][i] > 0)
				allID.push_back(idepth[lvl][i]);
		}
		std::sort(allID.begin(), allID.end());
		int n = allID.size()-1;

		float minID_new = allID[(int)(n*0.05)];
		float maxID_new = allID[(int)(n*0.95)];

		float minID, maxID;
		minID = minID_new;
		maxID = maxID_new;
		if(minID_pt!=0 && maxID_pt!=0)
		{
			if(*minID_pt < 0 || *maxID_pt < 0)
			{
				*maxID_pt = maxID;
				*minID_pt = minID;
			}
			else
			{

				// slowly adapt: change by maximum 10% of old span.
				float maxChange = 0.3*(*maxID_pt - *minID_pt);

				if(minID < *minID_pt - maxChange)
					minID = *minID_pt - maxChange;
				if(minID > *minID_pt + maxChange)
					minID = *minID_pt + maxChange;


				if(maxID < *maxID_pt - maxChange)
					maxID = *maxID_pt - maxChange;
				if(maxID > *maxID_pt + maxChange)
					maxID = *maxID_pt + maxChange;

				*maxID_pt = maxID;
				*minID_pt = minID;
			}
		}


		MinimalImageB3 mf(w[lvl], h[lvl]);
		mf.setBlack();
		for(int i=0;i<h[lvl]*w[lvl];i++)
		{
			int c = lastRef->dIp[lvl][i][0]*0.9f;
			if(c>255) c=255;
			mf.at(i) = Vec3b(c,c,c);
		}
		int wl = w[lvl];
		for(int y=3;y<h[lvl]-3;y++)
			for(int x=3;x<wl-3;x++)
			{
				int idx=x+y*wl;
				float sid=0, nid=0;
				float* bp = idepth[lvl]+idx;

				if(bp[0] > 0) {sid+=bp[0]; nid++;}
				if(bp[1] > 0) {sid+=bp[1]; nid++;}
				if(bp[-1] > 0) {sid+=bp[-1]; nid++;}
				if(bp[wl] > 0) {sid+=bp[wl]; nid++;}
				if(bp[-wl] > 0) {sid+=bp[-wl]; nid++;}

				if(bp[0] > 0 || nid >= 3)
				{
					float id = ((sid / nid)-minID) / ((maxID-minID));
					mf.setPixelCirc(x,y,makeJet3B(id));
					//mf.at(idx) = makeJet3B(id);
				}
			}
        //IOWrap::displayImage("coarseDepth LVL0", &mf, false);


        for(IOWrap::Output3DWrapper* ow : wraps)
            ow->pushDepthImage(&mf);

		if(debugSaveImages)
		{
			char buf[1000];
			snprintf(buf, 1000, "images_out/predicted_%05d_%05d.png", lastRef->shell->id, refFrameID);
			IOWrap::writeImage(buf,&mf);
		}

	}
}



void CoarseTracker::debugPlotIDepthMapFloat(std::vector<IOWrap::Output3DWrapper*> &wraps)
{
    if(w[1] == 0) return;
    int lvl = 0;
    MinimalImageF mim(w[lvl], h[lvl], idepth[lvl]);
    for(IOWrap::Output3DWrapper* ow : wraps)
        ow->pushDepthImageFloat(&mim, lastRef);
}











CoarseDistanceMap::CoarseDistanceMap(int ww, int hh)
{
	fwdWarpedIDDistFinal = new float[ww*hh/4];

	bfsList1 = new Eigen::Vector2i[ww*hh/4];
	bfsList2 = new Eigen::Vector2i[ww*hh/4];

	int fac = 1 << (pyrLevelsUsed-1);


	coarseProjectionGrid = new PointFrameResidual*[2048*(ww*hh/(fac*fac))];
	coarseProjectionGridNum = new int[ww*hh/(fac*fac)];

	w[0]=h[0]=0;
}
CoarseDistanceMap::~CoarseDistanceMap()
{
	delete[] fwdWarpedIDDistFinal;
	delete[] bfsList1;
	delete[] bfsList2;
	delete[] coarseProjectionGrid;
	delete[] coarseProjectionGridNum;
}




// @ 对于目前所有的地图点投影, 生成距离场图
void CoarseDistanceMap::makeDistanceMap(
		std::vector<FrameHessian*> frameHessians,
		FrameHessian* frame)
{
	int w1 = w[1];
	int h1 = h[1];
	int wh1 = w1*h1;
	for(int i=0;i<wh1;i++)
		fwdWarpedIDDistFinal[i] = 1000;


	// make coarse tracking templates for latstRef.
	int numItems = 0;

	for(FrameHessian* fh : frameHessians)
	{
		if(frame == fh) continue;

		SE3 fhToNew = frame->PRE_worldToCam * fh->PRE_camToWorld;
		Mat33f KRKi = (K[1] * fhToNew.rotationMatrix().cast<float>() * Ki[0]);
		Vec3f Kt = (K[1] * fhToNew.translation().cast<float>());

		for(PointHessian* ph : fh->pointHessians)
		{
			assert(ph->status == PointHessian::ACTIVE);
			Vec3f ptp = KRKi * Vec3f(ph->u, ph->v, 1) + Kt*ph->idepth_scaled;
			int u = ptp[0] / ptp[2] + 0.5f;
			int v = ptp[1] / ptp[2] + 0.5f;
			if(!(u > 0 && v > 0 && u < w[1] && v < h[1])) continue;
			fwdWarpedIDDistFinal[u+w1*v]=0;
			bfsList1[numItems] = Eigen::Vector2i(u,v);
			numItems++;
		}
	}

	growDistBFS(numItems);
}




void CoarseDistanceMap::makeInlierVotes(std::vector<FrameHessian*> frameHessians)
{

}



void CoarseDistanceMap::growDistBFS(int bfsNum)
{
	assert(w[0] != 0);
	int w1 = w[1], h1 = h[1];
	for(int k=1;k<40;k++)
	{
		int bfsNum2 = bfsNum;
		std::swap<Eigen::Vector2i*>(bfsList1,bfsList2);
		bfsNum=0;

		if(k%2==0)
		{
			for(int i=0;i<bfsNum2;i++)
			{
				int x = bfsList2[i][0];
				int y = bfsList2[i][1];
				if(x==0 || y== 0 || x==w1-1 || y==h1-1) continue;
				int idx = x + y * w1;

				if(fwdWarpedIDDistFinal[idx+1] > k)
				{
					fwdWarpedIDDistFinal[idx+1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x+1,y); bfsNum++;
				}
				if(fwdWarpedIDDistFinal[idx-1] > k)
				{
					fwdWarpedIDDistFinal[idx-1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x-1,y); bfsNum++;
				}
				if(fwdWarpedIDDistFinal[idx+w1] > k)
				{
					fwdWarpedIDDistFinal[idx+w1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x,y+1); bfsNum++;
				}
				if(fwdWarpedIDDistFinal[idx-w1] > k)
				{
					fwdWarpedIDDistFinal[idx-w1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x,y-1); bfsNum++;
				}
			}
		}
		else
		{
			for(int i=0;i<bfsNum2;i++)
			{
				int x = bfsList2[i][0];
				int y = bfsList2[i][1];
				if(x==0 || y== 0 || x==w1-1 || y==h1-1) continue;
				int idx = x + y * w1;

				if(fwdWarpedIDDistFinal[idx+1] > k)
				{
					fwdWarpedIDDistFinal[idx+1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x+1,y); bfsNum++;
				}
				if(fwdWarpedIDDistFinal[idx-1] > k)
				{
					fwdWarpedIDDistFinal[idx-1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x-1,y); bfsNum++;
				}
				if(fwdWarpedIDDistFinal[idx+w1] > k)
				{
					fwdWarpedIDDistFinal[idx+w1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x,y+1); bfsNum++;
				}
				if(fwdWarpedIDDistFinal[idx-w1] > k)
				{
					fwdWarpedIDDistFinal[idx-w1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x,y-1); bfsNum++;
				}

				if(fwdWarpedIDDistFinal[idx+1+w1] > k)
				{
					fwdWarpedIDDistFinal[idx+1+w1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x+1,y+1); bfsNum++;
				}
				if(fwdWarpedIDDistFinal[idx-1+w1] > k)
				{
					fwdWarpedIDDistFinal[idx-1+w1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x-1,y+1); bfsNum++;
				}
				if(fwdWarpedIDDistFinal[idx-1-w1] > k)
				{
					fwdWarpedIDDistFinal[idx-1-w1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x-1,y-1); bfsNum++;
				}
				if(fwdWarpedIDDistFinal[idx+1-w1] > k)
				{
					fwdWarpedIDDistFinal[idx+1-w1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x+1,y-1); bfsNum++;
				}
			}
		}
	}
}


void CoarseDistanceMap::addIntoDistFinal(int u, int v)
{
	if(w[0] == 0) return;
	bfsList1[0] = Eigen::Vector2i(u,v);
	fwdWarpedIDDistFinal[u+w[1]*v] = 0;
	growDistBFS(1);
}



void CoarseDistanceMap::makeK(CalibHessian* HCalib)
{
	w[0] = wG[0];
	h[0] = hG[0];

	fx[0] = HCalib->fxl();
	fy[0] = HCalib->fyl();
	cx[0] = HCalib->cxl();
	cy[0] = HCalib->cyl();

	for (int level = 1; level < pyrLevelsUsed; ++ level)
	{
		w[level] = w[0] >> level;
		h[level] = h[0] >> level;
		fx[level] = fx[level-1] * 0.5;
		fy[level] = fy[level-1] * 0.5;
		cx[level] = (cx[0] + 0.5) / ((int)1<<level) - 0.5;
		cy[level] = (cy[0] + 0.5) / ((int)1<<level) - 0.5;
	}

	for (int level = 0; level < pyrLevelsUsed; ++ level)
	{
		K[level]  << fx[level], 0.0, cx[level], 0.0, fy[level], cy[level], 0.0, 0.0, 1.0;
		Ki[level] = K[level].inverse();
		fxi[level] = Ki[level](0,0);
		fyi[level] = Ki[level](1,1);
		cxi[level] = Ki[level](0,2);
		cyi[level] = Ki[level](1,2);
	}
}

}
