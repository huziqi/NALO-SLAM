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

#include "FullSystem/FullSystem.h"
 
#include "stdio.h"
#include "util/globalFuncs.h"
#include <Eigen/LU>
#include <algorithm>
#include "IOWrapper/ImageDisplay.h"
#include "util/globalCalib.h"
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
#include "FullSystem/PixelSelector.h"
#include "FullSystem/PixelSelector2.h"
#include "FullSystem/ResidualProjections.h"
#include "FullSystem/ImmaturePoint.h"
#include "FullSystem/MapPoint.h"

#include "FullSystem/CoarseTracker.h"
#include "FullSystem/CoarseInitializer.h"

#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

#include "IOWrapper/Output3DWrapper.h"

#include "util/ImageAndExposure.h"

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <cmath>

using namespace std;
namespace dso
{
int FrameHessian::instanceCounter=0;
int PointHessian::instanceCounter=0;
int CalibHessian::instanceCounter=0;



FullSystem::FullSystem()
{

	int retstat =0;
	if(setting_logStuff)
	{

		retstat += system("rm -rf logs");
		retstat += system("mkdir logs");

		retstat += system("rm -rf mats");
		retstat += system("mkdir mats");

		calibLog = new std::ofstream();
		calibLog->open("logs/calibLog.txt", std::ios::trunc | std::ios::out);
		calibLog->precision(12);

		numsLog = new std::ofstream();
		numsLog->open("logs/numsLog.txt", std::ios::trunc | std::ios::out);
		numsLog->precision(10);

		coarseTrackingLog = new std::ofstream();
		coarseTrackingLog->open("logs/coarseTrackingLog.txt", std::ios::trunc | std::ios::out);
		coarseTrackingLog->precision(10);

		eigenAllLog = new std::ofstream();
		eigenAllLog->open("logs/eigenAllLog.txt", std::ios::trunc | std::ios::out);
		eigenAllLog->precision(10);

		eigenPLog = new std::ofstream();
		eigenPLog->open("logs/eigenPLog.txt", std::ios::trunc | std::ios::out);
		eigenPLog->precision(10);

		eigenALog = new std::ofstream();
		eigenALog->open("logs/eigenALog.txt", std::ios::trunc | std::ios::out);
		eigenALog->precision(10);

		DiagonalLog = new std::ofstream();
		DiagonalLog->open("logs/diagonal.txt", std::ios::trunc | std::ios::out);
		DiagonalLog->precision(10);

		variancesLog = new std::ofstream();
		variancesLog->open("logs/variancesLog.txt", std::ios::trunc | std::ios::out);
		variancesLog->precision(10);


		nullspacesLog = new std::ofstream();
		nullspacesLog->open("logs/nullspacesLog.txt", std::ios::trunc | std::ios::out);
		nullspacesLog->precision(10);
	}
	else
	{
		nullspacesLog=0;
		variancesLog=0;
		DiagonalLog=0;
		eigenALog=0;
		eigenPLog=0;
		eigenAllLog=0;
		numsLog=0;
		calibLog=0;
	}

	assert(retstat!=293847);



	selectionMap = new float[wG[0]*hG[0]];

	coarseDistanceMap = new CoarseDistanceMap(wG[0], hG[0]);
	coarseTracker = new CoarseTracker(wG[0], hG[0]);
	coarseTracker_forNewKF = new CoarseTracker(wG[0], hG[0]);
	coarseInitializer = new CoarseInitializer(wG[0], hG[0]);
	pixelSelector = new PixelSelector(wG[0], hG[0]);
	densemapper = new DenseMapping();
	densemapper->makeK(&Hcalib);


	statistics_lastNumOptIts=0;
	statistics_numDroppedPoints=0;
	statistics_numActivatedPoints=0;
	statistics_numCreatedPoints=0;
	statistics_numForceDroppedResBwd = 0;
	statistics_numForceDroppedResFwd = 0;
	statistics_numMargResFwd = 0;
	statistics_numMargResBwd = 0;

	lastCoarseRMSE.setConstant(100);

	currentMinActDist=2;
	initialized=false;


	ef = new EnergyFunctional();
	ef->red = &this->treadReduce;

	isLost=false;
	initFailed=false;


	needNewKFAfter = -1;

	linearizeOperation=true;
	runMapping=true;
	mappingThread = boost::thread(&FullSystem::mappingLoop, this);
	lastRefStopID=0;



	minIdJetVisDebug = -1;
	maxIdJetVisDebug = -1;
	minIdJetVisTracker = -1;
	maxIdJetVisTracker = -1;

	fix_num = -1;
	fix_count = 0;

	gplanefixed = false;

	acc_r = 0;

	tsdf.reset(new cpu_tsdf::TSDFVolumeOctree);
	tsdf->setGridSize (10., 10., 10.); // 10m x 10m x 10m
	tsdf->setResolution (2048, 2048, 2048); // Smallest cell size = 10m / 2048 = about half a centimeter
	tsdf->setIntegrateColor (false); // Set to true if you want the TSDF to store color
	// Eigen::Affine3d tsdf_center; // Optionally offset the center
	// tsdf->setGlobalTransform (tsdf_center);
	tsdf->reset (); // Initialize it to be empty
}

FullSystem::~FullSystem()
{
	blockUntilMappingIsFinished();

	if(setting_logStuff)
	{
		calibLog->close(); delete calibLog;
		numsLog->close(); delete numsLog;
		coarseTrackingLog->close(); delete coarseTrackingLog;
		//errorsLog->close(); delete errorsLog;
		eigenAllLog->close(); delete eigenAllLog;
		eigenPLog->close(); delete eigenPLog;
		eigenALog->close(); delete eigenALog;
		DiagonalLog->close(); delete DiagonalLog;
		variancesLog->close(); delete variancesLog;
		nullspacesLog->close(); delete nullspacesLog;
	}

	delete[] selectionMap;

	for(FrameShell* s : allFrameHistory)
		delete s;
	for(FrameHessian* fh : unmappedTrackedFrames)
		delete fh;

	delete coarseDistanceMap;
	delete coarseTracker;
	delete coarseTracker_forNewKF;
	delete densemapper;
	delete coarseInitializer;
	delete pixelSelector;
	delete ef;
}

void FullSystem::setOriginalCalib(const VecXf &originalCalib, int originalW, int originalH)
{

}

void FullSystem::setGammaFunction(float* BInv)
{
	if(BInv==0) return;

	// copy BInv.
	memcpy(Hcalib.Binv, BInv, sizeof(float)*256);


	// invert.
	for(int i=1;i<255;i++)
	{
		// find val, such that Binv[val] = i.
		// I dont care about speed for this, so do it the stupid way.

		for(int s=1;s<255;s++)
		{
			if(BInv[s] <= i && BInv[s+1] >= i)
			{
				Hcalib.B[i] = s+(i - BInv[s]) / (BInv[s+1]-BInv[s]);
				break;
			}
		}
	}
	Hcalib.B[0] = 0;
	Hcalib.B[255] = 255;
}

bool FrameSort(FrameShell *f1, FrameShell *f2) { return f1->timestamp < f2->timestamp; }

void smoothRescale(std::vector<float>& scales, int all_size)
{
	// int interval = 4, th = 0.01;
	// float front = 0, back = 0, f_diff = 0, b_diff = 0;

	// for (int i = interval; i < scales.size() - interval - 1; i++)
	// {
	// 	// 计算前后5个值的平均数
	// 	for (int j = 1; j <= interval;j++)
	// 	{
	// 		front += scales[i - j] / interval;
	// 		back  += scales[i + j] / interval;
	// 	}

	// 	f_diff = scales[i] - front;
	// 	b_diff = scales[i] - back;

	// 	// remove outlier
	// 	if(abs(f_diff)>th && abs(b_diff)>th && f_diff*b_diff>0) { scales[i] = scales[i - 1]; continue; }
	// 	// belong to back
	// 	if(abs(b_diff)<th) continue;
	// 	// belong to front
	// 	else { scales[i] = scales[i - 1]; continue; }
	// }


	// 将尺度改变的时刻硬数出来
	float ave_fix = 0;
	float diff_size = all_size - scales.size();

	//1
	for (int i = 0; i < 144 - diff_size;i++)
	{
		ave_fix += scales[i] / (144 - diff_size);
	}
	for (int i = 0; i < 144 - diff_size;i++)
		scales[i] = ave_fix;
	
	
	//2
	ave_fix = 0;
	for (int i = 144 - diff_size; i < 343 - diff_size; i++)
	{
		ave_fix += scales[i] / (199);
	}
	for (int i = 144 - diff_size; i < 343 - diff_size; i++)
		scales[i] = ave_fix;


	//3
	ave_fix = 0;
	for (int i = 343 - diff_size; i < 469 - diff_size; i++)
	{
		ave_fix += scales[i] / (126);
	}
	for (int i = 343 - diff_size; i < 469 - diff_size; i++)
		scales[i] = ave_fix;


	//4
	ave_fix = 0;
	for (int i = 469 - diff_size; i < 754 - diff_size; i++)
	{
		ave_fix += scales[i] / (285);
	}
	for (int i = 469 - diff_size; i < 754 - diff_size; i++)
		scales[i] = ave_fix;

	//5
	ave_fix = 0;
	for (int i = 754 - diff_size; i < 900 - diff_size; i++)
	{
		ave_fix += scales[i] / (146);
	}
	for (int i = 754 - diff_size; i < 900 - diff_size; i++)
		scales[i] = ave_fix;

	//6
	ave_fix = 0;
	for (int i = 900 - diff_size; i < 939 - diff_size; i++)
	{
		ave_fix += scales[i] / (39);
	}
	for (int i = 900 - diff_size; i < 939 - diff_size; i++)
		scales[i] = ave_fix;

	
	//7
	ave_fix = 0;
	for (int i = 939 - diff_size; i < all_size - diff_size; i++)
	{
		ave_fix += scales[i] / (all_size - 939);
	}
	for (int i = 939 - diff_size; i < all_size - diff_size; i++)
		scales[i] = ave_fix;


	// debug
	for (int i = 0; i < scales.size();i++)
	{
		printf("scales的大小：%f\n", scales[i]);
	}
}

void FullSystem::saveCloudfile(std::string file)
{
	boost::unique_lock<boost::mutex> lock(trackMutex);
	boost::unique_lock<boost::mutex> crlock(shellPoseMutex);

	std::ofstream myfile;
	myfile.open (file.c_str());
	myfile << std::setprecision(15);

	// sort keyframes by timestamps
	std::sort(allKeyFramesHistory.begin(), allKeyFramesHistory.end(), FrameSort);

	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr sum_cloud(new pcl::PointCloud<pcl::PointXYZRGBA>);

	int sum_size = 0;
	// for(FrameShell* s : allFrameHistory)
	for (int i = 0; i < allKeyFramesHistory.size();i++)
	{
		if(allKeyFramesHistory[i]->cloud->points.size()==0) continue;
		char pcd_name[50];
        sprintf(pcd_name, "/home/hzq/dso/pcd/%u.pcd", i);
		allKeyFramesHistory[i]->cloud->height = 1;
		allKeyFramesHistory[i]->cloud->width = allKeyFramesHistory[i]->cloud->points.size();
		printf("keyframe%d has %d points\n", i, allKeyFramesHistory[i]->cloud->points.size());
		pcl::io::savePCDFileASCII(pcd_name, *allKeyFramesHistory[i]->cloud);

		// write poses.txt
		FrameShell *s = allKeyFramesHistory[i];
		if (!s->poseValid)
		{
			if(i==0)
			{
				myfile  << 0 << " " << 0 << " " << 0 << "\n";
			}
			else
			{
				FrameShell *sb = allKeyFramesHistory[i - 1];
				myfile << sb->camToWorld.translation().transpose()<<"\n";
			}
			continue;
		}
	
		myfile << s->camToWorld.translation().transpose()<<"\n";

		// add to sum cloud
		Eigen::Matrix4d c2w(allKeyFramesHistory[i]->camToWorld.matrix());
		for (auto pt : allKeyFramesHistory[i]->cloud->points)
		{
			Eigen::Vector4d point;
			point << (double)pt.x, (double)pt.y, (double)pt.z, 1;
			point = c2w * point;

			pcl::PointXYZRGBA dst_point;
			Eigen::Vector4f point_ = point.cast<float>();
			dst_point.x = point_[0]*PCL_SCALE;
			dst_point.y = point_[1]*PCL_SCALE;
			dst_point.z = point_[2]*PCL_SCALE;
			dst_point.r = 255;
			dst_point.g = 255;
			dst_point.b = 255;

			sum_cloud->points.push_back(dst_point);
			sum_size++;
		}
	}
	myfile.close();

	sum_cloud->height = 1;
	sum_cloud->width = sum_size;
	pcl::io::savePCDFileASCII("/home/hzq/dso/sum_dso.pcd", *sum_cloud);
}

void FullSystem::printResult(std::string file)
{
	boost::unique_lock<boost::mutex> lock(trackMutex);
	boost::unique_lock<boost::mutex> crlock(shellPoseMutex);

	std::ofstream myfile;
	myfile.open (file.c_str());
	myfile << std::setprecision(15);

	// sort keyframes by timestamps
	std::sort(allFrameHistory.begin(), allFrameHistory.end(), FrameSort);

	//for(FrameShell* s : allFrameHistory)
	for (int i = 0; i < allFrameHistory.size();i++)
	{
		FrameShell *s = allFrameHistory[i];
		if (!s->poseValid)
		{
			if(i==0)
			{
				myfile << s->timestamp <<
						" " << 0<<
						" " << 0<<
						" " << 0<<
						" " << 0<<
						" " << 0<<
						" " << 0<<
						" " << 0 << "\n";
			}
			else
			{
				FrameShell *sb = allFrameHistory[i - 1];
				myfile << s->timestamp << " " << sb->camToWorld.translation().transpose() 
				<< " " << sb->camToWorld.so3().unit_quaternion().x() << " " 
				<< sb->camToWorld.so3().unit_quaternion().y() << " " 
				<< sb->camToWorld.so3().unit_quaternion().z() << " " 
				<< sb->camToWorld.so3().unit_quaternion().w() << "\n";
			}
			continue;
		}
			

		// if(setting_onlyLogKFPoses && s->marginalizedAt == s->id) continue;


		myfile << s->timestamp <<
			" " << s->camToWorld.translation().transpose()<<
			" " << s->camToWorld.so3().unit_quaternion().x()<<
			" " << s->camToWorld.so3().unit_quaternion().y()<<
			" " << s->camToWorld.so3().unit_quaternion().z()<<
			" " << s->camToWorld.so3().unit_quaternion().w() << "\n";

	}
	myfile.close();
}

//@ 使用确定的运动模型对新来的一帧进行跟踪, 得到位姿和光度参数
Vec4 FullSystem::trackNewCoarse(FrameHessian* fh)
{
	assert(allFrameHistory.size() > 0);
	// set pose initialization.

    for(IOWrap::Output3DWrapper* ow : outputWrapper)
        ow->pushLiveFrame(fh);



	FrameHessian* lastF = coarseTracker->lastRef;//参考帧

	AffLight aff_last_2_l = AffLight(0,0);
	//[ ***step 1*** ] 设置不同的运动状态
	std::vector<SE3,Eigen::aligned_allocator<SE3>> lastF_2_fh_tries;
	if(allFrameHistory.size() == 2)
		for(unsigned int i=0;i<lastF_2_fh_tries.size();i++) lastF_2_fh_tries.push_back(SE3());
	else
	{
		FrameShell* slast = allFrameHistory[allFrameHistory.size()-2];//上一帧
		FrameShell* sprelast = allFrameHistory[allFrameHistory.size()-3];//上上一帧
		SE3 slast_2_sprelast;
		SE3 lastF_2_slast;
		{	// lock on global pose consistency!
			boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
			slast_2_sprelast = sprelast->camToWorld.inverse() * slast->camToWorld;// 上一帧和上上一帧的运动
			lastF_2_slast = slast->camToWorld.inverse() * lastF->shell->camToWorld;// 参考帧到上一帧运动
			aff_last_2_l = slast->aff_g2l;
		}
		SE3 fh_2_slast = slast_2_sprelast;// assumed to be the same as fh_2_slast.当前帧到上一帧 = 上一帧和大上一帧的

		//! 尝试不同的运动模型
		// get last delta-movement.
		lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast);	// assume constant motion.
		lastF_2_fh_tries.push_back(fh_2_slast.inverse() * fh_2_slast.inverse() * lastF_2_slast);	// assume double motion (frame skipped)
		lastF_2_fh_tries.push_back(SE3::exp(fh_2_slast.log()*0.5).inverse() * lastF_2_slast); // assume half motion.
		lastF_2_fh_tries.push_back(lastF_2_slast); // assume zero motion.
		lastF_2_fh_tries.push_back(SE3()); // assume zero motion FROM KF.

		//! 尝试不同的旋转变动
		// just try a TON of different initializations (all rotations). In the end,
		// if they don't work they will only be tried on the coarsest level, which is super fast anyway.
		// also, if tracking rails here we loose, so we really, really want to avoid that.
		for(float rotDelta=0.02; rotDelta < 0.05; rotDelta++)
		{
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,0,0), Vec3(0,0,0)));			// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,rotDelta,0), Vec3(0,0,0)));			// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,0,rotDelta), Vec3(0,0,0)));			// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,0,0), Vec3(0,0,0)));			// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,-rotDelta,0), Vec3(0,0,0)));			// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,0,-rotDelta), Vec3(0,0,0)));			// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,0,rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,-rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,0,rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,-rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,0,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,-rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,-rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,0,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,-rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,-rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,-rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,-rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
		}

		if(!slast->poseValid || !sprelast->poseValid || !lastF->shell->poseValid)
		{
			lastF_2_fh_tries.clear();
			lastF_2_fh_tries.push_back(SE3());
		}
	}


	Vec3 flowVecs = Vec3(100,100,100);
	SE3 lastF_2_fh = SE3();
	AffLight aff_g2l = AffLight(0,0);


	//! as long as maxResForImmediateAccept is not reached, I'll continue through the options.
	//! I'll keep track of the so-far best achieved residual for each level in achievedRes. 
	//! 把到目前为止最好的残差值作为每一层的阈值
	//! If on a coarse level, tracking is WORSE than achievedRes, we will not continue to save time.
	//! 粗层的能量值大, 也不继续优化了, 来节省时间


	Vec5 achievedRes = Vec5::Constant(NAN);
	bool haveOneGood = false;
	int tryIterations=0;
	//! 逐个尝试
	for(unsigned int i=0;i<lastF_2_fh_tries.size();i++)
	{
		//[ ***step 2*** ] 尝试不同的运动状态, 得到跟踪是否良好
		AffLight aff_g2l_this = aff_last_2_l;
		SE3 lastF_2_fh_this = lastF_2_fh_tries[i];


		bool trackingIsGood = coarseTracker->trackNewestCoarse(
				fh, lastF_2_fh_this, aff_g2l_this,
				pyrLevelsUsed-1, achievedRes);	// in each level has to be at least as good as the last try.
		
		
		tryIterations++;

		if(false)//i!=0
		{
			printf("RE-TRACK ATTEMPT %d with initOption %d and start-lvl %d (ab %f %f): %f %f %f %f %f -> %f %f %f %f %f \n",
					i,
					i, pyrLevelsUsed-1,
					aff_g2l_this.a,aff_g2l_this.b,
					achievedRes[0],
					achievedRes[1],
					achievedRes[2],
					achievedRes[3],
					achievedRes[4],
					coarseTracker->lastResiduals[0],
					coarseTracker->lastResiduals[1],
					coarseTracker->lastResiduals[2],
					coarseTracker->lastResiduals[3],
					coarseTracker->lastResiduals[4]);
		}

		//[ ***step 3*** ] 如果跟踪正常, 并且0层残差比最好的还好留下位姿, 保存最好的每一层的能量值
		// do we have a new winner?
		if(trackingIsGood && std::isfinite((float)coarseTracker->lastResiduals[0]) && !(coarseTracker->lastResiduals[0] >=  achievedRes[0]))
		{
			//printf("take over. minRes %f -> %f!\n", achievedRes[0], coarseTracker->lastResiduals[0]);
			flowVecs = coarseTracker->lastFlowIndicators;
			aff_g2l = aff_g2l_this;
			lastF_2_fh = lastF_2_fh_this;
			haveOneGood = true;
		}

		// take over achieved res (always).
		if(haveOneGood)
		{
			for(int i=0;i<5;i++)
			{
				if(!std::isfinite((float)achievedRes[i]) || achievedRes[i] > coarseTracker->lastResiduals[i])	// take over if achievedRes is either bigger or NAN.
					achievedRes[i] = coarseTracker->lastResiduals[i];
			}
		}

		//[ ***step 4*** ] 小于阈值则暂停, 并且为下次设置阈值
        if(haveOneGood &&  achievedRes[0] < lastCoarseRMSE[0]*setting_reTrackThreshold)
            break;

	}

	if(!haveOneGood)
	{
        printf("BIG ERROR! tracking failed entirely. Take predictred pose and hope we may somehow recover.\n");
		flowVecs = Vec3(0,0,0);
		aff_g2l = aff_last_2_l;
		lastF_2_fh = lastF_2_fh_tries[0];
	}

	lastCoarseRMSE = achievedRes;

	// no lock required, as fh is not used anywhere yet.
	// 更新当前帧的位姿信息！！
	fh->shell->camToTrackingRef = lastF_2_fh.inverse();
	fh->shell->trackingRef = lastF->shell;
	fh->shell->aff_g2l = aff_g2l;
	fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;


	if(coarseTracker->firstCoarseRMSE < 0)
		coarseTracker->firstCoarseRMSE = achievedRes[0];

    if(!setting_debugout_runquiet)
        printf("Coarse Tracker tracked ab = %f %f (exp %f). Res %f!\n", aff_g2l.a, aff_g2l.b, fh->ab_exposure, achievedRes[0]);



	if(setting_logStuff)
	{
		(*coarseTrackingLog) << std::setprecision(16)
						<< fh->shell->id << " "
						<< fh->shell->timestamp << " "
						<< fh->ab_exposure << " "
						<< fh->shell->camToWorld.log().transpose() << " "
						<< aff_g2l.a << " "
						<< aff_g2l.b << " "
						<< achievedRes[0] << " "
						<< tryIterations << "\n";
	}


	return Vec4(achievedRes[0], flowVecs[0], flowVecs[1], flowVecs[2]);
}

//@ 利用新的帧 fh 对关键帧中的ImmaturePoint进行更新
void FullSystem::traceNewCoarse(FrameHessian* fh)
{
	boost::unique_lock<boost::mutex> lock(mapMutex);

	int trace_total=0, trace_good=0, trace_oob=0, trace_out=0, trace_skip=0, trace_badcondition=0, trace_uninitialized=0;

	Mat33f K = Mat33f::Identity();
	K(0,0) = Hcalib.fxl();
	K(1,1) = Hcalib.fyl();
	K(0,2) = Hcalib.cxl();
	K(1,2) = Hcalib.cyl();

	for(FrameHessian* host : frameHessians)		// go through all active frames
	{

		SE3 hostToNew = fh->PRE_worldToCam * host->PRE_camToWorld;
		Mat33f KRKi = K * hostToNew.rotationMatrix().cast<float>() * K.inverse();
		Vec3f Kt = K * hostToNew.translation().cast<float>();

		Vec2f aff = AffLight::fromToVecExposure(host->ab_exposure, fh->ab_exposure, host->aff_g2l(), fh->aff_g2l()).cast<float>();

		for(ImmaturePoint* ph : host->immaturePoints)
		{
			ph->traceOn(fh, KRKi, Kt, aff, &Hcalib, false );

			if(ph->lastTraceStatus==ImmaturePointStatus::IPS_GOOD) trace_good++;
			if(ph->lastTraceStatus==ImmaturePointStatus::IPS_BADCONDITION) trace_badcondition++;
			if(ph->lastTraceStatus==ImmaturePointStatus::IPS_OOB) trace_oob++;
			if(ph->lastTraceStatus==ImmaturePointStatus::IPS_OUTLIER) trace_out++;
			if(ph->lastTraceStatus==ImmaturePointStatus::IPS_SKIPPED) trace_skip++;
			if(ph->lastTraceStatus==ImmaturePointStatus::IPS_UNINITIALIZED) trace_uninitialized++;
			trace_total++;
		}
	}
//	printf("ADD: TRACE: %'d points. %'d (%.0f%%) good. %'d (%.0f%%) skip. %'d (%.0f%%) badcond. %'d (%.0f%%) oob. %'d (%.0f%%) out. %'d (%.0f%%) uninit.\n",
//			trace_total,
//			trace_good, 100*trace_good/(float)trace_total,
//			trace_skip, 100*trace_skip/(float)trace_total,
//			trace_badcondition, 100*trace_badcondition/(float)trace_total,
//			trace_oob, 100*trace_oob/(float)trace_total,
//			trace_out, 100*trace_out/(float)trace_total,
//			trace_uninitialized, 100*trace_uninitialized/(float)trace_total);
}




void FullSystem::activatePointsMT_Reductor(
		std::vector<PointHessian*>* optimized,
		std::vector<ImmaturePoint*>* toOptimize,
		int min, int max, Vec10* stats, int tid)
{
	ImmaturePointTemporaryResidual* tr = new ImmaturePointTemporaryResidual[frameHessians.size()];
	for(int k=min;k<max;k++)
	{
		(*optimized)[k] = optimizeImmaturePoint((*toOptimize)[k],1,tr);
	}
	delete[] tr;
}



void FullSystem::activatePointsMT()
{
	//[ ***step 1*** ] 阈值计算, 通过距离地图来控制数目
	if(ef->nPoints < setting_desiredPointDensity*0.66)
		currentMinActDist -= 0.8;
	if(ef->nPoints < setting_desiredPointDensity*0.8)
		currentMinActDist -= 0.5;
	else if(ef->nPoints < setting_desiredPointDensity*0.9)
		currentMinActDist -= 0.2;
	else if(ef->nPoints < setting_desiredPointDensity)
		currentMinActDist -= 0.1;

	if(ef->nPoints > setting_desiredPointDensity*1.5)
		currentMinActDist += 0.8;
	if(ef->nPoints > setting_desiredPointDensity*1.3)
		currentMinActDist += 0.5;
	if(ef->nPoints > setting_desiredPointDensity*1.15)
		currentMinActDist += 0.2;
	if(ef->nPoints > setting_desiredPointDensity)
		currentMinActDist += 0.1;

	if(currentMinActDist < 0) currentMinActDist = 0;
	if(currentMinActDist > 4) currentMinActDist = 4;

    if(!setting_debugout_runquiet)
        printf("SPARSITY:  MinActDist %f (need %d points, have %d points)!\n",
                currentMinActDist, (int)(setting_desiredPointDensity), ef->nPoints);



	FrameHessian* newestHs = frameHessians.back();

	// make dist map.
	coarseDistanceMap->makeK(&Hcalib);
	coarseDistanceMap->makeDistanceMap(frameHessians, newestHs);

	//coarseTracker->debugPlotDistMap("distMap");

	std::vector<ImmaturePoint*> toOptimize; toOptimize.reserve(20000);

	//[ ***step 2*** ] 处理未成熟点, 激活/删除/跳过
	for(FrameHessian* host : frameHessians)		// go through all active frames
	{
		if(host == newestHs) continue;

		SE3 fhToNew = newestHs->PRE_worldToCam * host->PRE_camToWorld;
		Mat33f KRKi = (coarseDistanceMap->K[1] * fhToNew.rotationMatrix().cast<float>() * coarseDistanceMap->Ki[0]);
		Vec3f Kt = (coarseDistanceMap->K[1] * fhToNew.translation().cast<float>());


		for(unsigned int i=0;i<host->immaturePoints.size();i+=1)
		{
			ImmaturePoint* ph = host->immaturePoints[i];
			ph->idxInImmaturePoints = i;

			// delete points that have never been traced successfully, or that are outlier on the last trace.
			if(!std::isfinite(ph->idepth_max) || ph->lastTraceStatus == IPS_OUTLIER)
			{
				// immature_invalid_deleted++;
				// remove point.
				delete ph;
				host->immaturePoints[i]=0;
				continue;
			}

			// can activate only if this is true.
			bool canActivate = (ph->lastTraceStatus == IPS_GOOD
							 || ph->lastTraceStatus == IPS_SKIPPED
							 || ph->lastTraceStatus == IPS_BADCONDITION
							 || ph->lastTraceStatus == IPS_OOB )
						&& ph->lastTracePixelInterval < 8
						&& ph->quality > setting_minTraceQuality
						&& (ph->idepth_max+ph->idepth_min) > 0;


			// if I cannot activate the point, skip it. Maybe also delete it.？？？？？可能不需要删除！！！！！
			if(!canActivate)
			{
				// if point will be out afterwards, delete it instead.
				if(ph->host->flaggedForMarginalization || ph->lastTraceStatus == IPS_OOB)
				{
					// immature_notReady_deleted++;
					delete ph;
					host->immaturePoints[i]=0;
				}
					// immature_notReady_skipped++;
				continue;
			}


			// see if we need to activate point due to distance map.
			Vec3f ptp = KRKi * Vec3f(ph->u, ph->v, 1) + Kt*(0.5f*(ph->idepth_max+ph->idepth_min));
			int u = ptp[0] / ptp[2] + 0.5f;
			int v = ptp[1] / ptp[2] + 0.5f;

			if((u > 0 && v > 0 && u < wG[1] && v < hG[1]))
			{

				float dist = coarseDistanceMap->fwdWarpedIDDistFinal[u+wG[1]*v] + (ptp[0]-floorf((float)(ptp[0])));

				if(dist>=currentMinActDist* ph->my_type)
				{
					coarseDistanceMap->addIntoDistFinal(u,v);
					toOptimize.push_back(ph);
				}
			}
			else
			{
				delete ph;
				host->immaturePoints[i]=0;
			}
		}
	}


//	printf("ACTIVATE: %d. (del %d, notReady %d, marg %d, good %d, marg-skip %d)\n",
//			(int)toOptimize.size(), immature_deleted, immature_notReady, immature_needMarg, immature_want, immature_margskip);

	//[ ***step 3*** ] 优化上一步挑出来的未成熟点, 进行逆深度优化, 并得到pointhessian
	std::vector<PointHessian*> optimized; optimized.resize(toOptimize.size());

	if(multiThreading)
		treadReduce.reduce(boost::bind(&FullSystem::activatePointsMT_Reductor, this, &optimized, &toOptimize, _1, _2, _3, _4), 0, toOptimize.size(), 50);

	else
		activatePointsMT_Reductor(&optimized, &toOptimize, 0, toOptimize.size(), 0, 0);


	//[ ***step 4*** ] 把PointHessian加入到能量函数, 删除不收敛的未成熟点, 或不好的点
	for(unsigned k=0;k<toOptimize.size();k++)
	{
		PointHessian* newpoint = optimized[k];
		ImmaturePoint* ph = toOptimize[k];

		if(newpoint != 0 && newpoint != (PointHessian*)((long)(-1)))
		{
			newpoint->host->immaturePoints[ph->idxInImmaturePoints]=0;
			newpoint->host->pointHessians.push_back(newpoint);
			ef->insertPoint(newpoint);
			for(PointFrameResidual* r : newpoint->residuals)
				ef->insertResidual(r);
			assert(newpoint->efPoint != 0);
			delete ph;
		}
		else if(newpoint == (PointHessian*)((long)(-1)) || ph->lastTraceStatus==IPS_OOB)
		{
			ph->host->immaturePoints[ph->idxInImmaturePoints]=0;
			delete ph;
		}
		else
		{
			assert(newpoint == 0 || newpoint == (PointHessian*)((long)(-1)));
		}
	}


	for(FrameHessian* host : frameHessians)
	{
		for(int i=0;i<(int)host->immaturePoints.size();i++)
		{
			if(host->immaturePoints[i]==0)
			{
				host->immaturePoints[i] = host->immaturePoints.back();
				host->immaturePoints.pop_back();
				i--;
			}
		}
	}


}


void FullSystem::flagPointsForRemoval()
{
	assert(EFIndicesValid);

	std::vector<FrameHessian*> fhsToKeepPoints;
	std::vector<FrameHessian*> fhsToMargPoints;

	//if(setting_margPointVisWindow>0)
	{
		for(int i=((int)frameHessians.size())-1;i>=0 && i >= ((int)frameHessians.size());i--)
			if(!frameHessians[i]->flaggedForMarginalization) fhsToKeepPoints.push_back(frameHessians[i]);

		for(int i=0; i< (int)frameHessians.size();i++)
			if(frameHessians[i]->flaggedForMarginalization) fhsToMargPoints.push_back(frameHessians[i]);
	}



	// ef->setAdjointsF();
	// ef->setDeltaF(&Hcalib);
	int flag_oob=0, flag_in=0, flag_inin=0, flag_nores=0;

	for(FrameHessian* host : frameHessians)		// go through all active frames
	{
		for(unsigned int i=0;i<host->pointHessians.size();i++)
		{
			PointHessian* ph = host->pointHessians[i];
			if(ph==0) continue;

			if(ph->idepth_scaled < 0 || ph->residuals.size()==0)
			{
				host->pointHessiansOut.push_back(ph);
				ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
				host->pointHessians[i]=0;
				flag_nores++;
			}
			else if(ph->isOOB(fhsToKeepPoints, fhsToMargPoints) || host->flaggedForMarginalization)
			{
				flag_oob++;
				if(ph->isInlierNew())
				{
					flag_in++;
					int ngoodRes=0;
					for(PointFrameResidual* r : ph->residuals)
					{
						r->resetOOB();
						r->linearize(&Hcalib);
						r->efResidual->isLinearized = false;
						r->applyRes(true);
						if(r->efResidual->isActive())
						{
							r->efResidual->fixLinearizationF(ef);
							ngoodRes++;
						}
					}
                    if(ph->idepth_hessian > setting_minIdepthH_marg)
					{
						flag_inin++;
						ph->efPoint->stateFlag = EFPointStatus::PS_MARGINALIZE;
						host->pointHessiansMarginalized.push_back(ph);
					}
					else
					{
						ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
						host->pointHessiansOut.push_back(ph);
					}


				}
				else
				{
					host->pointHessiansOut.push_back(ph);
					ph->efPoint->stateFlag = EFPointStatus::PS_DROP;


					//printf("drop point in frame %d (%d goodRes, %d activeRes)\n", ph->host->idx, ph->numGoodResiduals, (int)ph->residuals.size());
				}

				host->pointHessians[i]=0;
			}
		}


		for(int i=0;i<(int)host->pointHessians.size();i++)
		{
			if(host->pointHessians[i]==0)
			{
				host->pointHessians[i] = host->pointHessians.back();
				host->pointHessians.pop_back();
				i--;
			}
		}
	}

}


void FullSystem::addActiveFrame( ImageAndExposure* image, int id )
{

    if(isLost) return;
	boost::unique_lock<boost::mutex> lock(trackMutex);//跟踪线程锁


	// =========================== add into allFrameHistory =========================
	FrameHessian* fh = new FrameHessian();//图像帧的影像信息
	FrameShell* shell = new FrameShell();//图像帧的位姿信息
	shell->camToWorld = SE3(); 		// no lock required, as fh is not used anywhere yet.
	shell->aff_g2l = AffLight(0,0);
    shell->marginalizedAt = shell->id = allFrameHistory.size();
    shell->timestamp = image->timestamp;
    shell->incoming_id = id;
	shell->cloud.reset(new pcl::PointCloud<pcl::PointXYZRGBA>);
	
	fh->shell = shell;

	fh->mask = new float[wG[0] * hG[0]];
	fh->img_bgr = new Vec3b[wG[0] * hG[0]];
	for (int i = 0; i < wG[0] * hG[0]; i++)
	{
		fh->mask[i] = image->mask[i];
		fh->img_bgr[i] = image->img_bgr[i];
	}

	allFrameHistory.push_back(shell);

	// =========================== make Images / derivatives etc. =========================
	fh->ab_exposure = image->exposure_time;
    fh->makeImages(image->image, &Hcalib);//构建图像金字塔


	if(!initialized)
	{
		// use initializer!
		if(coarseInitializer->frameID<0)	// first frame set. fh is kept by coarseInitializer.
		{

			coarseInitializer->setFirst(&Hcalib, fh);
		}
		else if(coarseInitializer->trackFrame(fh, outputWrapper))	// if SNAPPED
		{

			initializeFromInitializer(fh);
			lock.unlock();
			deliverTrackedFrame(fh, true);
		}
		else
		{
			// if still initializing
			fh->shell->poseValid = false;
			delete fh;
		}
		return;
	}
	else	// do front-end operation.
	{
		// =========================== SWAP tracking reference?. =========================
		if(coarseTracker_forNewKF->refFrameID > coarseTracker->refFrameID)
		{
			boost::unique_lock<boost::mutex> crlock(coarseTrackerSwapMutex);
			CoarseTracker* tmp = coarseTracker; coarseTracker=coarseTracker_forNewKF; coarseTracker_forNewKF=tmp;
		}

		// 优化位姿，使用旋转和位移对像素移动的作用比来判断运动状态
		Vec4 tres = trackNewCoarse(fh);

		// std::cout << "trackNewCoarse后frame"<<fh->shell->id<<"的位姿："<<std::endl<<"[" << fh->shell->camToWorld.matrix3x4()<<"]" << std::endl;

		if(!std::isfinite((double)tres[0]) || !std::isfinite((double)tres[1]) || !std::isfinite((double)tres[2]) || !std::isfinite((double)tres[3]))
        {
            printf("Initial Tracking failed: LOST!\n");
			isLost=true;
            return;
        }

		// 判断是否插入关键帧
		bool needToMakeKF = false;
		if(setting_keyframesPerSecond > 0) // 每隔多久插入关键帧
		{
			needToMakeKF = allFrameHistory.size()== 1 ||
					(fh->shell->timestamp - allKeyFramesHistory.back()->timestamp) > 0.95f/setting_keyframesPerSecond;
		}
		else
		{
			Vec2 refToFh=AffLight::fromToVecExposure(coarseTracker->lastRef->ab_exposure, fh->ab_exposure,
					coarseTracker->lastRef_aff_g2l, fh->shell->aff_g2l);

			// BRIGHTNESS CHECK
			needToMakeKF = allFrameHistory.size()== 1 ||
					setting_kfGlobalWeight*setting_maxShiftWeightT *  sqrtf((double)tres[1]) / (wG[0]+hG[0]) +  // 平移像素位移
					setting_kfGlobalWeight*setting_maxShiftWeightR *  sqrtf((double)tres[2]) / (wG[0]+hG[0]) + 	// 旋转像素位移, 设置为0???
					setting_kfGlobalWeight*setting_maxShiftWeightRT * sqrtf((double)tres[3]) / (wG[0]+hG[0]) +	// 旋转+平移像素位移
					setting_kfGlobalWeight*setting_maxAffineWeight * fabs(logf((float)refToFh[0])) > 1 ||		// 光度变化大
					2*coarseTracker->firstCoarseRMSE < tres[0];		// 误差能量变化太大(最初的两倍)

		}

        for(IOWrap::Output3DWrapper* ow : outputWrapper)
            ow->publishCamPose(fh->shell, &Hcalib);

		
		lock.unlock();
		deliverTrackedFrame(fh, needToMakeKF);// 把当前帧喂给建图线程
		return;
	}
}
//@ 把跟踪的帧传给建图线程
void FullSystem::deliverTrackedFrame(FrameHessian* fh, bool needKF)
{
	if(linearizeOperation)
	{
		//printf("一步一步运行\n");
		if (goStepByStep && lastRefStopID != coarseTracker->refFrameID)
		{
			MinimalImageB3 img(wG[0], hG[0], fh->img_bgr);
			IOWrap::displayImage("frameToTrack", &img);
			while(true)
			{
				char k=IOWrap::waitKey(0);
				if(k==' ') break;
				handleKey( k );
			}
			lastRefStopID = coarseTracker->refFrameID;
		}
		else handleKey( IOWrap::waitKey(1) );

		if (needKF)
			makeKeyFrame(fh);
		else makeNonKeyFrame(fh);
	}
	else
	{
		boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);
		unmappedTrackedFrames.push_back(fh);
		if(needKF) needNewKFAfter=fh->shell->trackingRef->id;
		trackedFrameSignal.notify_all();

		while(coarseTracker_forNewKF->refFrameID == -1 && coarseTracker->refFrameID == -1 )
		{
			mappedFrameSignal.wait(lock);
		}

		lock.unlock();
	}
}

void FullSystem::mappingLoop()
{
	boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);

	while(runMapping)
	{
		while(unmappedTrackedFrames.size()==0)
		{
			trackedFrameSignal.wait(lock);
			if(!runMapping) return;
		}

		FrameHessian* fh = unmappedTrackedFrames.front();
		unmappedTrackedFrames.pop_front();


		// guaranteed to make a KF for the very first two tracked frames.
		if(allKeyFramesHistory.size() <= 2)
		{
			lock.unlock();
			makeKeyFrame(fh);
			lock.lock();
			mappedFrameSignal.notify_all();
			continue;
		}

		if(unmappedTrackedFrames.size() > 3)
			needToKetchupMapping=true;


		if(unmappedTrackedFrames.size() > 0) // if there are other frames to tracke, do that first.
		{
			lock.unlock();
			makeNonKeyFrame(fh);
			lock.lock();

			if(needToKetchupMapping && unmappedTrackedFrames.size() > 0)
			{
				FrameHessian* fh = unmappedTrackedFrames.front();
				unmappedTrackedFrames.pop_front();
				{
					boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
					assert(fh->shell->trackingRef != 0);
					fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
					fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(),fh->shell->aff_g2l);
				}
				delete fh;
			}

		}
		else
		{
			if(setting_realTimeMaxKF || needNewKFAfter >= frameHessians.back()->shell->id)
			{
				lock.unlock();
				makeKeyFrame(fh);
				needToKetchupMapping=false;
				lock.lock();
			}
			else
			{
				lock.unlock();
				makeNonKeyFrame(fh);
				lock.lock();
			}
		}
		mappedFrameSignal.notify_all();
	}
	printf("MAPPING FINISHED!\n");
}

void FullSystem::blockUntilMappingIsFinished()
{
	boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);
	runMapping = false;
	trackedFrameSignal.notify_all();
	lock.unlock();

	mappingThread.join();

}

void FullSystem::makeNonKeyFrame( FrameHessian* fh)
{
	// needs to be set by mapping thread. no lock required since we are in mapping thread.
	{
		boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
		assert(fh->shell->trackingRef != 0);
		fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
		fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(),fh->shell->aff_g2l);
	}

	traceNewCoarse(fh);
	delete fh;
}

void FullSystem::makeKeyFrame( FrameHessian* fh)
{
	// needs to be set by mapping thread
	{
		boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
		assert(fh->shell->trackingRef != 0);
		fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
		fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(),fh->shell->aff_g2l);//设置相机位姿尺度

		// =========================== 计算尺度比例 ===============================

		// if(scale_fix)
		// {
		// 	// assert(coarseTracker_forNewKF->rScale > 0 && std::isfinite(coarseTracker_forNewKF->rScale));

		// 	if(fix_num==-1)
		// 		fix_count=fix_num = fh->shell->id;
		// 	float rscale = coarseTracker_forNewKF->rScale;
		// 	//ReScale.push_back(rscale);

		// 	fh->shell->camToWorld.translation() *= rscale;

		// 	printf("尺度固定在第%ld帧，第%d帧的尺度比为：%f\n", fix_num,fix_count++,rscale);
		// }

		//fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(),fh->shell->aff_g2l);//设置相机位姿尺度
	}

	/*//=============================
	留下的点(去除边缘化+删除)占所有点小于5%和参考帧比曝光变化较大
	保证滑窗内有5个关键帧
	如果还大于7个关键帧,则边缘化掉到最新关键帧的距离占所有距离比最大的。保证良好的空间结构
	*/

	// 计算未成熟点的逆深度范围
	traceNewCoarse(fh);

	boost::unique_lock<boost::mutex> lock(mapMutex);

	// =========================== Flag Frames to be Marginalized. =========================
	flagFramesForMarginalization(fh);


	// =========================== add New Frame to Hessian Struct. =========================
	fh->idx = frameHessians.size();
	frameHessians.push_back(fh);
	fh->frameID = allKeyFramesHistory.size();
	allKeyFramesHistory.push_back(fh->shell);
	ef->insertFrame(fh, &Hcalib);

	setPrecalcValues();



	// =========================== add new residuals for old points =========================
	int numFwdResAdde=0;
	for(FrameHessian* fh1 : frameHessians)		// go through all active frames
	{
		if(fh1 == fh) continue;
		for(PointHessian* ph : fh1->pointHessians)
		{
			PointFrameResidual* r = new PointFrameResidual(ph, fh1, fh);
			r->setState(ResState::IN);
			ph->residuals.push_back(r);
			ef->insertResidual(r);
			ph->lastResiduals[1] = ph->lastResiduals[0];
			ph->lastResiduals[0] = std::pair<PointFrameResidual*, ResState>(r, ResState::IN);
			numFwdResAdde+=1;
		}
	}




	// =========================== Activate Points (& flag for marginalization). =========================
	activatePointsMT();
	ef->makeIDX();



	// =========================== OPTIMIZE ALL =========================

	fh->frameEnergyTH = frameHessians.back()->frameEnergyTH;
	float rmse = optimize(setting_maxOptIterations);


	
		


	// =========================== Figure Out if INITIALIZATION FAILED =========================
	if(allKeyFramesHistory.size() <= 4)
	{
		if(allKeyFramesHistory.size()==2 && rmse > 20*benchmark_initializerSlackFactor)
		{
			printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
			initFailed=true;
		}
		if(allKeyFramesHistory.size()==3 && rmse > 13*benchmark_initializerSlackFactor)
		{
			printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
			initFailed=true;
		}
		if(allKeyFramesHistory.size()==4 && rmse > 9*benchmark_initializerSlackFactor)
		{
			printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
			initFailed=true;
		}
	}



    if(isLost) return;




	// =========================== REMOVE OUTLIER =========================
	removeOutliers();



	{
		boost::unique_lock<boost::mutex> crlock(coarseTrackerSwapMutex);
		coarseTracker_forNewKF->makeK(&Hcalib);
		coarseTracker_forNewKF->setCoarseTrackingRef(frameHessians); // 把最新帧设置为参考帧



        coarseTracker_forNewKF->debugPlotIDepthMap(&minIdJetVisTracker, &maxIdJetVisTracker, outputWrapper);
        coarseTracker_forNewKF->debugPlotIDepthMapFloat(outputWrapper);
	}

	debugPlot("post Optimize");

// ========================== PLANE PARAMETER OPTIMIZE ==============
	// printf("frame%d ground--->[%lf, %lf, %lf, %lf]\n", frameHessians.back()->shell->id, frameHessians.back()->groundP[0],
																						// frameHessians.back()->groundP[1],
																						// frameHessians.back()->groundP[2],
																						// frameHessians.back()->groundP[3]);
	// if(frameHessians.back()->haveground&&setPlaneOptimize)//kitti_code
	if(frameHessians.back()->haveground&&setPlaneOptimize&&frameHessians.back()->groundP[3]!=0)
	{
		// printf("第%d帧有%d个地面\n", frameHessians[frameHessians.size()-1]->shell->id, frameHessians[frameHessians.size()-1]->haveground);
		if(!gplanefixed)
			gplanefixed = setglobalplane(frameHessians);
		
		// if(!gplanefixed)	
		// 	gplanefixed = setinitgroundheight(frameHessians);

		// 计算累计旋转
		// Vec3 r_win0 = frameHessians[0]->PRE_camToWorld.log().tail<3>();
		// Vec3 r_win1 = frameHessians.back()->PRE_camToWorld.log().tail<3>();
		// acc_r = abs(r_win0.squaredNorm() - r_win1.squaredNorm());

		// cout << "!!!!!!!!!!!!!!!!acc_r: " << acc_r << endl;
		// if(acc_r>2)
		// 	resetGlobalPlane(frameHessians);

		if(scale_fix&&gplanefixed)
		{
			float prmse = planeOptimize(Hcalib, frameHessians);
			SWGrayOptimize_J(frameHessians);
		}
	}

	// =========================== (Activate-)Marginalize Points =========================
	flagPointsForRemoval();
	ef->dropPointsF();
	getNullspaces(
			ef->lastNullspaces_pose,
			ef->lastNullspaces_scale,
			ef->lastNullspaces_affA,
			ef->lastNullspaces_affB);
	ef->marginalizePointsF();



	// =========================== add new Immature points & new residuals =========================
	makeNewTraces(fh, 0);


	// ==============恢复局部稠密地图============

	// display color images
	// MinimalImageB3 color_img(wG[0],hG[0]);
	// for(int i=0;i<wG[0]*hG[0];i++)
	// {
	// 	float b = frameHessians.back()->img_bgr[i][0]*1.0;
	// 	float g = frameHessians.back()->img_bgr[i][1]*1.0;
	// 	float r = frameHessians.back()->img_bgr[i][2]*1.0;
	// 	if(b>255) b=255;
	// 	if(g>255) g=255;
	// 	if(r>255) r=255;
	// 	color_img.at(i) = Vec3b(b,g,r);
	// }
	// IOWrap::displayImage("color image", &color_img);

	// display pmask
	// MinimalImageB pmask(wG[0],hG[0]);
	// for(int i=0;i<wG[0]*hG[0];i++)
	// {
	// 	float b = frameHessians.back()->pmask[i] * 1.0;

	// 	if(b>255) b=255;
	// 	pmask.at(i) = b;
	// }
	// IOWrap::displayImage("pmask", &pmask);

	if(gplanefixed&&denseMapping)
	{
		cout << "###############   frame[" << fh->frameID << "]    ###################" << endl;
		
		// 取倒数第三帧作为局部稠密恢复的关键帧
		FrameHessian *dfh = frameHessians[frameHessians.size() - 3];
		densemapper->updateMap(frameHessians,dfh);
		// densemapper->refineMap(frameHessians, dfh);
	}

	for (IOWrap::Output3DWrapper *ow : outputWrapper)
	{
		ow->publishGraph(ef->connectivityMap);
		ow->publishKeyframes(frameHessians, false, &Hcalib);
	}

	// =========================== Marginalize Frames =========================

	for(unsigned int i=0;i<frameHessians.size();i++)
		if(frameHessians[i]->flaggedForMarginalization)
			{marginalizeFrame(frameHessians[i]); i=0;}

	// if(setting_tsdf)
	// {
	// 	for(auto fh:frameHessians)
	// 	{
	// 		fh->shell->cloud->points.clear();
	// 		for (auto pt : fh->pointHessians)
	// 		{
	// 			for(int pnt=0;pnt<patternNum;pnt++)
	// 			{
	// 				pcl::PointXYZRGBA point;
	// 				int dx = patternP[pnt][0];
	// 				int dy = patternP[pnt][1];
	// 				point.x = (((pt->u+dx) * fxiG[0] + cxiG[0]) / pt->idepth_scaled) * PCL_SCALE;
	// 				point.y = (((pt->v+dy) * fyiG[0] + cyiG[0]) / pt->idepth_scaled) * PCL_SCALE;
	// 				point.z = ((1 + 2 * fxiG[0] * (rand() / (float)RAND_MAX - 0.5f)) / pt->idepth_scaled) * PCL_SCALE;
	// 				point.r = 255;
	// 				point.g = 255;
	// 				point.b = 255;

	// 				if (std::isnan(point.x) || std::isnan(point.y) || std::isnan(point.z))
	// 					continue;
	// 				fh->shell->cloud->points.push_back(point);
	// 			}
				
	// 		}
	// 		for(auto pt:fh->immaturePoints)
	// 		{
	// 			for(int pnt=0;pnt<patternNum;pnt++)
	// 			{
	// 				int dx = patternP[pnt][0];
	// 				int dy = patternP[pnt][1];
	// 				pcl::PointXYZRGBA point;
	// 				point.x = (((pt->u+dx) * fxiG[0] + cxiG[0]) / ((pt->idepth_max + pt->idepth_min) * 0.5f)) * PCL_SCALE;
	// 				point.y = (((pt->v+dy) * fyiG[0] + cyiG[0]) / ((pt->idepth_max + pt->idepth_min) * 0.5f)) * PCL_SCALE;
	// 				point.z = ((1 + 2 * fxiG[0] * (rand() / (float)RAND_MAX - 0.5f)) / ((pt->idepth_max + pt->idepth_min) * 0.5f)) * PCL_SCALE;
	// 				point.r = 255;
	// 				point.g = 255;
	// 				point.b = 255;
	// 				if(std::isnan(point.x)||std::isnan(point.y)||std::isnan(point.z)) continue;
	// 				fh->shell->cloud->points.push_back(point);
	// 			}
				
	// 		}

	// 		// Eigen::Translation3d trans(fh->shell->camToWorld.translation().x(),
	// 		// 						   fh->shell->camToWorld.translation().y(), fh->shell->camToWorld.translation().z());
	// 		// Eigen::Affine3d pose(trans);
	// 		// tsdf->integrateCloud(*cloud, pcl::PointCloud<pcl::Normal>(), pose);
	// 	}
	// }

	printLogLine();
    //printEigenValLine();

}

//@ 从初始化中提取出信息, 用于跟踪.
void FullSystem::initializeFromInitializer(FrameHessian* newFrame)
{
	boost::unique_lock<boost::mutex> lock(mapMutex);

	//[ ***step 1*** ] 把第一帧设置成关键帧, 加入队列, 加入EnergyFunctional
	// add firstframe.
	FrameHessian* firstFrame = coarseInitializer->firstFrame;  // 第一帧增加进地图
	firstFrame->idx = frameHessians.size(); // 赋值给它id (0开始)
	frameHessians.push_back(firstFrame);  	// 地图内关键帧容器
	firstFrame->frameID = allKeyFramesHistory.size();  	// 所有历史关键帧id
	allKeyFramesHistory.push_back(firstFrame->shell); 	// 所有历史关键帧
	ef->insertFrame(firstFrame, &Hcalib);
	setPrecalcValues();   		// 设置相对位姿预计算值

	//int numPointsTotal = makePixelStatus(firstFrame->dI, selectionMap, wG[0], hG[0], setting_desiredDensity);
	//int numPointsTotal = pixelSelector->makeMaps(firstFrame->dIp, selectionMap,setting_desiredDensity);

	firstFrame->pointHessians.reserve(wG[0]*hG[0]*0.2f);
	firstFrame->pointHessiansMarginalized.reserve(wG[0]*hG[0]*0.2f);
	firstFrame->pointHessiansOut.reserve(wG[0]*hG[0]*0.2f);

	//[ ***step 2*** ] 求出平均尺度因子
	float sumID=1e-5, numID=1e-5;
	for(int i=0;i<coarseInitializer->numPoints[0];i++)
	{
		sumID += coarseInitializer->points[0][i].iR;
		numID++;
	}
	float rescaleFactor = 1 / (sumID / numID);

	// randomly sub-select the points I need.
	float keepPercentage = setting_desiredPointDensity / coarseInitializer->numPoints[0];

    if(!setting_debugout_runquiet)
        printf("Initialization: keep %.1f%% (need %d, have %d)!\n", 100*keepPercentage,
                (int)(setting_desiredPointDensity), coarseInitializer->numPoints[0] );

	//[ ***step 3*** ] 创建PointHessian, 点加入关键帧, 加入EnergyFunctional
	for(int i=0;i<coarseInitializer->numPoints[0];i++)
	{
		if(rand()/(float)RAND_MAX > keepPercentage) continue;

		Pnt* point = coarseInitializer->points[0]+i;
		ImmaturePoint* pt = new ImmaturePoint(point->u+0.5f,point->v+0.5f,firstFrame,point->my_type, &Hcalib);

		if(!std::isfinite(pt->energyTH)) { delete pt; continue; }


		pt->idepth_max=pt->idepth_min=1;
		PointHessian* ph = new PointHessian(pt, &Hcalib);
		delete pt;
		if(!std::isfinite(ph->energyTH)) {delete ph; continue;}

		ph->setIdepthScaled(point->iR*rescaleFactor);
		ph->setIdepthZero(ph->idepth);
		ph->hasDepthPrior=true;
		ph->setPointStatus(PointHessian::ACTIVE);

		firstFrame->pointHessians.push_back(ph);
		ef->insertPoint(ph);
	}



	SE3 firstToNew = coarseInitializer->thisToNext;
	firstToNew.translation() /= rescaleFactor;

	//[ ***step 4*** ] 设置第一帧和最新帧的待优化量, 参考帧
	// really no lock required, as we are initializing.
	{
		boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
		firstFrame->shell->camToWorld = SE3();
		firstFrame->shell->aff_g2l = AffLight(0,0);
		firstFrame->setEvalPT_scaled(firstFrame->shell->camToWorld.inverse(),firstFrame->shell->aff_g2l);
		firstFrame->shell->trackingRef=0;
		firstFrame->shell->camToTrackingRef = SE3();

		newFrame->shell->camToWorld = firstToNew.inverse();
		newFrame->shell->aff_g2l = AffLight(0,0);
		newFrame->setEvalPT_scaled(newFrame->shell->camToWorld.inverse(),newFrame->shell->aff_g2l);
		newFrame->shell->trackingRef = firstFrame->shell;
		newFrame->shell->camToTrackingRef = firstToNew.inverse();

	}

	initialized=true;
	printf("INITIALIZE FROM INITIALIZER (%d pts)!\n", (int)firstFrame->pointHessians.size());
}

void FullSystem::makeNewTraces(FrameHessian* newFrame, float* gtDepth)
{
	pixelSelector->allowFast = true;
	int numPointsTotal=0;
	if (!setting_useLidar)
	{
		//int numPointsTotal = makePixelStatus(newFrame->dI, selectionMap, wG[0], hG[0], setting_desiredDensity);
		numPointsTotal = pixelSelector->makeMaps(newFrame, selectionMap,setting_desiredImmatureDensity);
	}
	else
	{
		//using lidar mask as pixel selection strategy
		numPointsTotal = pixelSelector->makeMaps_lidar(newFrame, selectionMap, setting_desiredImmatureDensity, 1, true, 1);
	}
	
	newFrame->pointHessians.reserve(numPointsTotal * 1.2f);
	//fh->pointHessiansInactive.reserve(numPointsTotal*1.2f);
	newFrame->pointHessiansMarginalized.reserve(numPointsTotal*1.2f);
	newFrame->pointHessiansOut.reserve(numPointsTotal*1.2f);


	for(int y=patternPadding+1;y<hG[0]-patternPadding-2;y++)
	for(int x=patternPadding+1;x<wG[0]-patternPadding-2;x++)
	{
		int i = x+y*wG[0];
		if(selectionMap[i]==0) continue;

		ImmaturePoint* impt = new ImmaturePoint(x,y,newFrame, selectionMap[i], &Hcalib);
		if(!std::isfinite(impt->energyTH)) delete impt;
		else newFrame->immaturePoints.push_back(impt);

	}
	//printf("MADE %d IMMATURE POINTS!\n", (int)newFrame->immaturePoints.size());

}



void FullSystem::setPrecalcValues()
{
	for(FrameHessian* fh : frameHessians)
	{
		fh->targetPrecalc.resize(frameHessians.size());
		for(unsigned int i=0;i<frameHessians.size();i++)
			fh->targetPrecalc[i].set(fh, frameHessians[i], &Hcalib);
	}

	ef->setDeltaF(&Hcalib);
}


void FullSystem::printLogLine()
{
	if(frameHessians.size()==0) return;

    if(!setting_debugout_runquiet)
        printf("LOG %d: %.3f fine. Res: %d A, %d L, %d M; (%'d / %'d) forceDrop. a=%f, b=%f. Window %d (%d)\n",
                allKeyFramesHistory.back()->id,
                statistics_lastFineTrackRMSE,
                ef->resInA,
                ef->resInL,
                ef->resInM,
                (int)statistics_numForceDroppedResFwd,
                (int)statistics_numForceDroppedResBwd,
                allKeyFramesHistory.back()->aff_g2l.a,
                allKeyFramesHistory.back()->aff_g2l.b,
                frameHessians.back()->shell->id - frameHessians.front()->shell->id,
                (int)frameHessians.size());


	if(!setting_logStuff) return;

	if(numsLog != 0)
	{
		(*numsLog) << allKeyFramesHistory.back()->id << " "  <<
				statistics_lastFineTrackRMSE << " "  <<
				(int)statistics_numCreatedPoints << " "  <<
				(int)statistics_numActivatedPoints << " "  <<
				(int)statistics_numDroppedPoints << " "  <<
				(int)statistics_lastNumOptIts << " "  <<
				ef->resInA << " "  <<
				ef->resInL << " "  <<
				ef->resInM << " "  <<
				statistics_numMargResFwd << " "  <<
				statistics_numMargResBwd << " "  <<
				statistics_numForceDroppedResFwd << " "  <<
				statistics_numForceDroppedResBwd << " "  <<
				frameHessians.back()->aff_g2l().a << " "  <<
				frameHessians.back()->aff_g2l().b << " "  <<
				frameHessians.back()->shell->id - frameHessians.front()->shell->id << " "  <<
				(int)frameHessians.size() << " "  << "\n";
		numsLog->flush();
	}


}



void FullSystem::printEigenValLine()
{
	if(!setting_logStuff) return;
	if(ef->lastHS.rows() < 12) return;


	MatXX Hp = ef->lastHS.bottomRightCorner(ef->lastHS.cols()-CPARS,ef->lastHS.cols()-CPARS);
	MatXX Ha = ef->lastHS.bottomRightCorner(ef->lastHS.cols()-CPARS,ef->lastHS.cols()-CPARS);
	int n = Hp.cols()/8;
	assert(Hp.cols()%8==0);

	// sub-select
	for(int i=0;i<n;i++)
	{
		MatXX tmp6 = Hp.block(i*8,0,6,n*8);
		Hp.block(i*6,0,6,n*8) = tmp6;

		MatXX tmp2 = Ha.block(i*8+6,0,2,n*8);
		Ha.block(i*2,0,2,n*8) = tmp2;
	}
	for(int i=0;i<n;i++)
	{
		MatXX tmp6 = Hp.block(0,i*8,n*8,6);
		Hp.block(0,i*6,n*8,6) = tmp6;

		MatXX tmp2 = Ha.block(0,i*8+6,n*8,2);
		Ha.block(0,i*2,n*8,2) = tmp2;
	}

	VecX eigenvaluesAll = ef->lastHS.eigenvalues().real();
	VecX eigenP = Hp.topLeftCorner(n*6,n*6).eigenvalues().real();
	VecX eigenA = Ha.topLeftCorner(n*2,n*2).eigenvalues().real();
	VecX diagonal = ef->lastHS.diagonal();

	std::sort(eigenvaluesAll.data(), eigenvaluesAll.data()+eigenvaluesAll.size());
	std::sort(eigenP.data(), eigenP.data()+eigenP.size());
	std::sort(eigenA.data(), eigenA.data()+eigenA.size());

	int nz = std::max(100,setting_maxFrames*10);

	if(eigenAllLog != 0)
	{
		VecX ea = VecX::Zero(nz); ea.head(eigenvaluesAll.size()) = eigenvaluesAll;
		(*eigenAllLog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
		eigenAllLog->flush();
	}
	if(eigenALog != 0)
	{
		VecX ea = VecX::Zero(nz); ea.head(eigenA.size()) = eigenA;
		(*eigenALog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
		eigenALog->flush();
	}
	if(eigenPLog != 0)
	{
		VecX ea = VecX::Zero(nz); ea.head(eigenP.size()) = eigenP;
		(*eigenPLog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
		eigenPLog->flush();
	}

	if(DiagonalLog != 0)
	{
		VecX ea = VecX::Zero(nz); ea.head(diagonal.size()) = diagonal;
		(*DiagonalLog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
		DiagonalLog->flush();
	}

	if(variancesLog != 0)
	{
		VecX ea = VecX::Zero(nz); ea.head(diagonal.size()) = ef->lastHS.inverse().diagonal();
		(*variancesLog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
		variancesLog->flush();
	}

	std::vector<VecX> &nsp = ef->lastNullspaces_forLogging;
	(*nullspacesLog) << allKeyFramesHistory.back()->id << " ";
	for(unsigned int i=0;i<nsp.size();i++)
		(*nullspacesLog) << nsp[i].dot(ef->lastHS * nsp[i]) << " " << nsp[i].dot(ef->lastbS) << " " ;
	(*nullspacesLog) << "\n";
	nullspacesLog->flush();

}

void FullSystem::printFrameLifetimes()
{
	if(!setting_logStuff) return;


	boost::unique_lock<boost::mutex> lock(trackMutex);

	std::ofstream* lg = new std::ofstream();
	lg->open("logs/lifetimeLog.txt", std::ios::trunc | std::ios::out);
	lg->precision(15);

	for(FrameShell* s : allFrameHistory)
	{
		(*lg) << s->id
			<< " " << s->marginalizedAt
			<< " " << s->statistics_goodResOnThis
			<< " " << s->statistics_outlierResOnThis
			<< " " << s->movedByOpt;



		(*lg) << "\n";
	}





	lg->close();
	delete lg;

}


void FullSystem::printEvalLine()
{
	return;
}

// 设置初始地面高度
bool FullSystem::setinitgroundheight(std::vector<FrameHessian *> fhs)
{	
	if(fhs.size()<=3)
		return false;
	double sum = 0;
	int count = 0;
	for (int i = 0; i < fhs.size();i++)
	{
		if (fhs[i]->haveground)
		{
			Eigen::Vector4d pih;

			pih << fhs[0]->groundP[0], fhs[0]->groundP[1], fhs[0]->groundP[2], fhs[0]->groundP[3];

			gplanefixed = true;

			//set local ground plane height
			if(pih[3]!=0)
			{
				sum += pih[3];
				count++;
			}
	
		}
	}
	if(count<2)
		return false;

	lgh = sum / count;
	printf("local ground height fixed at: %f\n", lgh);
	return true;
}

// 初始化全局地面参数
bool FullSystem::setglobalplane(std::vector<FrameHessian *> fhs)
{
	if (fhs.size() < setting_maxFrames)
		return false;

	int winsize = fhs.size();
	double sumnorm = 0;
	float sumd = 0;
	Eigen::Vector4d lastpi;

	lastpi << fhs[winsize-2]->groundP[0], fhs[winsize-2]->groundP[1], fhs[winsize-2]->groundP[2], fhs[winsize-2]->groundP[3];
	sumd += fhs[winsize - 2]->groundP[3];

	for (int i = winsize - 2; i > 0; i--)
	{
		Eigen::Vector4d pi, diffpi;
		pi = lastpi;
		lastpi[0] = fhs[i - 1]->groundP[0];
		lastpi[1] = fhs[i - 1]->groundP[1];
		lastpi[2] = fhs[i - 1]->groundP[2];
		lastpi[3] = fhs[i - 1]->groundP[3];

		// cout << "pi:" << pi.transpose() << endl;
		if (pi[3] == 0 || std::isnan(pi[3]) || std::isnan(pi[0]) || std::isnan(pi[1]) || std::isnan(pi[2]) || abs(pi[1]) > 1)
		{
			return false;
		}
		diffpi = pi - lastpi;
		sumnorm += diffpi.lpNorm<2>();

		sumd += lastpi[3];
	}
	printf("sumnorm: %f\n", sumnorm);
	if(sumnorm<0.2)//kitti_code
	// if(sumnorm<5)
	{
		// 将平面转换到世界坐标系下
		// pi'=(h2w^-1)^T*pi
		Eigen::Vector4d pih, piw;

		pih << fhs[1]->groundP[0], fhs[1]->groundP[1], fhs[1]->groundP[2], fhs[1]->groundP[3];//kitti_code
		// pih << fhs.back()->groundP[0], fhs.back()->groundP[1], fhs.back()->groundP[2], fhs.back()->groundP[3];

		piw = fhs[1]->PRE_worldToCam.matrix().transpose() * pih;//kitti_code
		// piw = fhs.back()->PRE_worldToCam.matrix().transpose() * pih;

		gplane[0] = piw[0];
		gplane[1] = piw[1];
		gplane[2] = piw[2];
		gplane[3] = piw[3];

		backup_gplane[0] = piw[0];
		backup_gplane[1] = piw[1];
		backup_gplane[2] = piw[2];
		backup_gplane[3] = piw[3];

		gplanefixed = true;

		//set local ground plane height
		lgh = pih[3];

		printf("global plane para fixed at:[%lf, %lf, %lf, %lf]\nlocal ground height fixed at: %f\n", gplane[0], gplane[1], gplane[2], gplane[3], lgh);
		return true;
	}
	return false;
}

// 重新初始化全局平面参数
void FullSystem::resetGlobalPlane(std::vector<FrameHessian *> fhs)
{
	int nwin = fhs.size();
	for (int i = nwin - 2; i >= 0; i--)
	{
		if(fhs[i]->haveground)
		{
			Vec4 tp;
			tp << fhs[i]->groundP[0], fhs[i]->groundP[1], fhs[i]->groundP[2], fhs[i]->groundP[3];

			tp = fhs[i]->PRE_worldToCam.matrix().transpose() * tp;

			gplane[0] = tp[0];
			gplane[1] = tp[1];
			gplane[2] = tp[2];
			gplane[3] = tp[3];

			printf("global plane reset at:[%lf, %lf, %lf, %lf]\n", gplane[0], gplane[1], gplane[2], gplane[3]);

			break;
		}
	}
}


}
