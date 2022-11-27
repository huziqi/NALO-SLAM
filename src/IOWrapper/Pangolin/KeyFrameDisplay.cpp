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



#include <stdio.h>
#include "util/settings.h"

//#include <GL/glx.h>
//#include <GL/gl.h>
//#include <GL/glu.h>

#include <pangolin/pangolin.h>
#include "KeyFrameDisplay.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/ImmaturePoint.h"
#include "util/FrameShell.h"



namespace dso
{
namespace IOWrap
{


KeyFrameDisplay::KeyFrameDisplay()
{
	originalInputSparse = 0;
	numSparseBufferSize=0;
	numSparsePoints=0;

	originalInputDense = 0;
	numDensePoints = 0;
	numDenseBufferSize = 0;

	id = 0;
	active= true;
	camToWorld = SE3();

	needRefresh=true;

	my_scaledTH =1e10;
	my_absTH = 1e10;
	my_displayMode = 0;
	my_minRelBS = 0;
	my_sparsifyFactor = 1;

	numGLBufferPoints=0;
	dnumGLBufferPoints=0;

	bufferValid = false;
	dbufferValid = false;
}
void KeyFrameDisplay::setFromF(FrameShell* frame, CalibHessian* HCalib)
{
	id = frame->id;
	fx = HCalib->fxl();
	fy = HCalib->fyl();
	cx = HCalib->cxl();
	cy = HCalib->cyl();
	width = wG[0];
	height = hG[0];
	fxi = 1/fx;
	fyi = 1/fy;
	cxi = -cx / fx;
	cyi = -cy / fy;
	camToWorld = frame->camToWorld;
	needRefresh=true;
}

void KeyFrameDisplay::setFromKF(FrameHessian* fh, CalibHessian* HCalib)
{
	setFromF(fh->shell, HCalib);

	// add all traces, inlier and outlier points.
	int npoints = fh->immaturePoints.size() +
				  fh->pointHessians.size() +
				  fh->pointHessiansMarginalized.size() +
				  fh->pointHessiansOut.size();
	int mpoints = fh->mapPoints.size();

	// printf("npoints: %d, mapPoints: %d\n", npoints, fh->mapPoints.size());

	if(numSparseBufferSize < npoints)
	{
		if(originalInputSparse != 0) delete originalInputSparse;
		numSparseBufferSize = npoints+100;
        originalInputSparse = new InputPointSparse<MAX_RES_PER_POINT>[numSparseBufferSize];
	}

	if(numDenseBufferSize<mpoints)
	{
		if(originalInputDense!=0) delete originalInputDense;
		numDenseBufferSize = mpoints + 100;
		originalInputDense = new InputDenseMap[numDenseBufferSize];
	}

	InputPointSparse<MAX_RES_PER_POINT>* pc = originalInputSparse;
	numSparsePoints = 0;
	for (ImmaturePoint *p : fh->immaturePoints)
	{
		for(int i=0;i<patternNum;i++)
			pc[numSparsePoints].color[i] = p->color[i];

		pc[numSparsePoints].u = p->u;
		pc[numSparsePoints].v = p->v;
		pc[numSparsePoints].idpeth = (p->idepth_max+p->idepth_min)*0.5f;
		pc[numSparsePoints].idepth_hessian = 1000;
		pc[numSparsePoints].relObsBaseline = 0;
		pc[numSparsePoints].numGoodRes = 1;
		pc[numSparsePoints].status = 0;
		numSparsePoints++;
	}

	for(PointHessian* p : fh->pointHessians)
	{
		for(int i=0;i<patternNum;i++)
			pc[numSparsePoints].color[i] = p->color[i];
		pc[numSparsePoints].u = p->u;
		pc[numSparsePoints].v = p->v;
		pc[numSparsePoints].idpeth = p->idepth_scaled;
		pc[numSparsePoints].relObsBaseline = p->maxRelBaseline;
		pc[numSparsePoints].idepth_hessian = p->idepth_hessian;
		pc[numSparsePoints].numGoodRes =  0;
		pc[numSparsePoints].status=1;

		numSparsePoints++;
	}

	for(PointHessian* p : fh->pointHessiansMarginalized)
	{
		for(int i=0;i<patternNum;i++)
			pc[numSparsePoints].color[i] = p->color[i];
		pc[numSparsePoints].u = p->u;
		pc[numSparsePoints].v = p->v;
		pc[numSparsePoints].idpeth = p->idepth_scaled;
		pc[numSparsePoints].relObsBaseline = p->maxRelBaseline;
		pc[numSparsePoints].idepth_hessian = p->idepth_hessian;
		pc[numSparsePoints].numGoodRes =  0;
		pc[numSparsePoints].status=2;
		numSparsePoints++;
	}

	for(PointHessian* p : fh->pointHessiansOut)
	{
		for(int i=0;i<patternNum;i++)
			pc[numSparsePoints].color[i] = p->color[i];
		pc[numSparsePoints].u = p->u;
		pc[numSparsePoints].v = p->v;
		pc[numSparsePoints].idpeth = p->idepth_scaled;
		pc[numSparsePoints].relObsBaseline = p->maxRelBaseline;
		pc[numSparsePoints].idepth_hessian = p->idepth_hessian;
		pc[numSparsePoints].numGoodRes =  0;
		pc[numSparsePoints].status=3;
		numSparsePoints++;
	}


	InputDenseMap *mpc = originalInputDense;
	numDensePoints = 0;
	for(MapPoint* p : fh->mapPoints)
	{
		mpc[numDensePoints].color[0] = p->bgr[0];
		mpc[numDensePoints].color[1] = p->bgr[1];
		mpc[numDensePoints].color[2] = p->bgr[2];

		mpc[numDensePoints].u = p->u;
		mpc[numDensePoints].v = p->v;
		mpc[numDensePoints].idpeth = p->idepth;

		numDensePoints++;
	}

	// printf("numsparse: %d, npoints: %d\n", numSparsePoints, npoints);
	assert(numSparsePoints <= npoints);
	assert(numDensePoints <= mpoints);

	camToWorld = fh->PRE_camToWorld;
	needRefresh=true;
}


KeyFrameDisplay::~KeyFrameDisplay()
{
	if(originalInputSparse != 0)
		delete[] originalInputSparse;
	if(originalInputDense!=0)
		delete[] originalInputDense;
}

void KeyFrameDisplay::refreshPC()
{
	// if there are no vertices, done!
	if(numDensePoints == 0)
		return;

	// make data
	Vec3f* tmpVertexBuffer = new Vec3f[numDensePoints];
	Vec3b* tmpColorBuffer = new Vec3b[numDensePoints];
	int vertexBufferNumPoints=0;

	int right = 0;

	for(int i=0;i<numDensePoints;i++)
	{
		if(originalInputDense[i].idpeth < 0) continue;


		float depth = 1.0f / originalInputDense[i].idpeth;

		tmpVertexBuffer[vertexBufferNumPoints][0] = ((originalInputDense[i].u)*fxi + cxi) * depth;
		tmpVertexBuffer[vertexBufferNumPoints][1] = ((originalInputDense[i].v)*fyi + cyi) * depth;
		tmpVertexBuffer[vertexBufferNumPoints][2] = depth*(1 + 2*fxi * (rand()/(float)RAND_MAX-0.5f));

		if(originalInputDense[i].u>wG[0]/2)
			right++;

		tmpColorBuffer[vertexBufferNumPoints][0] = originalInputDense[i].color[2];
		tmpColorBuffer[vertexBufferNumPoints][1] = originalInputDense[i].color[1];
		tmpColorBuffer[vertexBufferNumPoints][2] = originalInputDense[i].color[0];
		
		vertexBufferNumPoints++;


		assert(vertexBufferNumPoints <= numDensePoints);
		
	}

	// printf("right points:%d!!!!!!!!\n", right);

	if(vertexBufferNumPoints==0)
	{
		delete[] tmpColorBuffer;
		delete[] tmpVertexBuffer;
		return;
	}

	dnumGLBufferGoodPoints = vertexBufferNumPoints;
	if(dnumGLBufferGoodPoints > dnumGLBufferPoints)
	{
		dnumGLBufferPoints = vertexBufferNumPoints*1.3;
		dvertexBuffer.Reinitialise(pangolin::GlArrayBuffer, dnumGLBufferPoints, GL_FLOAT, 3, GL_DYNAMIC_DRAW );
		dcolorBuffer.Reinitialise(pangolin::GlArrayBuffer, dnumGLBufferPoints, GL_UNSIGNED_BYTE, 3, GL_DYNAMIC_DRAW );
	}
	dvertexBuffer.Upload(tmpVertexBuffer, sizeof(float)*3*dnumGLBufferGoodPoints, 0);
	dcolorBuffer.Upload(tmpColorBuffer, sizeof(unsigned char)*3*dnumGLBufferGoodPoints, 0);
	dbufferValid=true;
	delete[] tmpColorBuffer;
	delete[] tmpVertexBuffer;
}

bool KeyFrameDisplay::refreshPC(bool canRefresh, float scaledTH, float absTH, int mode, float minBS, int sparsity)
{
	if(canRefresh)
	{
		needRefresh = needRefresh ||
				my_scaledTH != scaledTH ||
				my_absTH != absTH ||
				my_displayMode != mode ||
				my_minRelBS != minBS ||
				my_sparsifyFactor != sparsity;
	}

	if(!needRefresh) return false;
	needRefresh=false;

	my_scaledTH = scaledTH;
	my_absTH = absTH;
	my_displayMode = mode;
	my_minRelBS = minBS;
	my_sparsifyFactor = sparsity;


	// if there are no vertices, done!
	if(numSparsePoints == 0)
		return false;

	// make data
	Vec3f* tmpVertexBuffer = new Vec3f[numSparsePoints*patternNum];
	Vec3b* tmpColorBuffer = new Vec3b[numSparsePoints*patternNum];
	int vertexBufferNumPoints=0;

	for(int i=0;i<numSparsePoints;i++)
	{
		/* display modes:
		 * my_displayMode==0 - all pts, color-coded
		 * my_displayMode==1 - normal points
		 * my_displayMode==2 - active only
		 * my_displayMode==3 - nothing
		 */

		if(my_displayMode==1 && originalInputSparse[i].status != 1 && originalInputSparse[i].status!= 2) continue;
		if(my_displayMode==2 && originalInputSparse[i].status != 1) continue;
		if(my_displayMode>2) continue;

		if(originalInputSparse[i].idpeth < 0) continue;


		float depth = 1.0f / originalInputSparse[i].idpeth;
		float depth4 = depth*depth; depth4*= depth4;
		float var = (1.0f / (originalInputSparse[i].idepth_hessian+0.01));

		if(var * depth4 > my_scaledTH)
			continue;

		if(var > my_absTH)
			continue;

		if(originalInputSparse[i].relObsBaseline < my_minRelBS)
			continue;


		for(int pnt=0;pnt<patternNum;pnt++)
		{

			if(my_sparsifyFactor > 1 && rand()%my_sparsifyFactor != 0) continue;
			int dx = patternP[pnt][0];
			int dy = patternP[pnt][1];

			tmpVertexBuffer[vertexBufferNumPoints][0] = ((originalInputSparse[i].u+dx)*fxi + cxi) * depth;
			tmpVertexBuffer[vertexBufferNumPoints][1] = ((originalInputSparse[i].v+dy)*fyi + cyi) * depth;
			tmpVertexBuffer[vertexBufferNumPoints][2] = depth*(1 + 2*fxi * (rand()/(float)RAND_MAX-0.5f));



			if(my_displayMode==0)
			{
				if(originalInputSparse[i].status==0)
				{
					tmpColorBuffer[vertexBufferNumPoints][0] = 0;
					tmpColorBuffer[vertexBufferNumPoints][1] = 255;
					tmpColorBuffer[vertexBufferNumPoints][2] = 255;
				}
				else if(originalInputSparse[i].status==1)
				{
					tmpColorBuffer[vertexBufferNumPoints][0] = 0;
					tmpColorBuffer[vertexBufferNumPoints][1] = 255;
					tmpColorBuffer[vertexBufferNumPoints][2] = 0;
				}
				else if(originalInputSparse[i].status==2)
				{
					tmpColorBuffer[vertexBufferNumPoints][0] = 0;
					tmpColorBuffer[vertexBufferNumPoints][1] = 0;
					tmpColorBuffer[vertexBufferNumPoints][2] = 255;
				}
				else if(originalInputSparse[i].status==3)
				{
					tmpColorBuffer[vertexBufferNumPoints][0] = 255;
					tmpColorBuffer[vertexBufferNumPoints][1] = 0;
					tmpColorBuffer[vertexBufferNumPoints][2] = 0;
				}
				else
				{
					tmpColorBuffer[vertexBufferNumPoints][0] = 255;
					tmpColorBuffer[vertexBufferNumPoints][1] = 255;
					tmpColorBuffer[vertexBufferNumPoints][2] = 255;
				}

			}
			else
			{
				tmpColorBuffer[vertexBufferNumPoints][0] = originalInputSparse[i].color[pnt];
				tmpColorBuffer[vertexBufferNumPoints][1] = originalInputSparse[i].color[pnt];
				tmpColorBuffer[vertexBufferNumPoints][2] = originalInputSparse[i].color[pnt];
			}
			vertexBufferNumPoints++;


			assert(vertexBufferNumPoints <= numSparsePoints*patternNum);
		}
	}

	if(vertexBufferNumPoints==0)
	{
		delete[] tmpColorBuffer;
		delete[] tmpVertexBuffer;
		return true;
	}

	numGLBufferGoodPoints = vertexBufferNumPoints;
	if(numGLBufferGoodPoints > numGLBufferPoints)
	{
		numGLBufferPoints = vertexBufferNumPoints*1.3;
		vertexBuffer.Reinitialise(pangolin::GlArrayBuffer, numGLBufferPoints, GL_FLOAT, 3, GL_DYNAMIC_DRAW );
		colorBuffer.Reinitialise(pangolin::GlArrayBuffer, numGLBufferPoints, GL_UNSIGNED_BYTE, 3, GL_DYNAMIC_DRAW );
	}
	vertexBuffer.Upload(tmpVertexBuffer, sizeof(float)*3*numGLBufferGoodPoints, 0);
	colorBuffer.Upload(tmpColorBuffer, sizeof(unsigned char)*3*numGLBufferGoodPoints, 0);
	bufferValid=true;
	delete[] tmpColorBuffer;
	delete[] tmpVertexBuffer;


	return true;
}

bool KeyFrameDisplay::refreshPC(bool canRefresh, float scaledTH, float absTH, int mode, float minBS, int sparsity, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
	if(canRefresh)
	{
		needRefresh = needRefresh ||
				my_scaledTH != scaledTH ||
				my_absTH != absTH ||
				my_displayMode != mode ||
				my_minRelBS != minBS ||
				my_sparsifyFactor != sparsity;
	}

	if(!needRefresh) return false;
	needRefresh=false;

	my_scaledTH = scaledTH;
	my_absTH = absTH;
	my_displayMode = mode;
	my_minRelBS = minBS;
	my_sparsifyFactor = sparsity;


	// if there are no vertices, done!
	if(numSparsePoints == 0)
		return false;

	// make data
	Vec3f* tmpVertexBuffer = new Vec3f[numSparsePoints*patternNum];
	Vec3b* tmpColorBuffer = new Vec3b[numSparsePoints*patternNum];
	int vertexBufferNumPoints=0;

	for(int i=0;i<numSparsePoints;i++)
	{
		/* display modes:
		 * my_displayMode==0 - all pts, color-coded
		 * my_displayMode==1 - normal points
		 * my_displayMode==2 - active only
		 * my_displayMode==3 - nothing
		 */

		if(my_displayMode==1 && originalInputSparse[i].status != 1 && originalInputSparse[i].status!= 2) continue;
		if(my_displayMode==2 && originalInputSparse[i].status != 1) continue;
		if(my_displayMode>2) continue;

		if(originalInputSparse[i].idpeth < 0) continue;


		float depth = 1.0f / originalInputSparse[i].idpeth;
		float depth4 = depth*depth; depth4*= depth4;
		float var = (1.0f / (originalInputSparse[i].idepth_hessian+0.01));

		if(var * depth4 > my_scaledTH)
			continue;

		if(var > my_absTH)
			continue;

		if(originalInputSparse[i].relObsBaseline < my_minRelBS)
			continue;


		for(int pnt=0;pnt<patternNum;pnt++)
		{

			if(my_sparsifyFactor > 1 && rand()%my_sparsifyFactor != 0) continue;
			int dx = patternP[pnt][0];
			int dy = patternP[pnt][1];

			tmpVertexBuffer[vertexBufferNumPoints][0] = ((originalInputSparse[i].u+dx)*fxi + cxi) * depth;
			tmpVertexBuffer[vertexBufferNumPoints][1] = ((originalInputSparse[i].v+dy)*fyi + cyi) * depth;
			tmpVertexBuffer[vertexBufferNumPoints][2] = depth*(1 + 2*fxi * (rand()/(float)RAND_MAX-0.5f));



			if(my_displayMode==0)
			{
				if(originalInputSparse[i].status==0)
				{
					tmpColorBuffer[vertexBufferNumPoints][0] = 0;
					tmpColorBuffer[vertexBufferNumPoints][1] = 255;
					tmpColorBuffer[vertexBufferNumPoints][2] = 255;
				}
				else if(originalInputSparse[i].status==1)
				{
					tmpColorBuffer[vertexBufferNumPoints][0] = 0;
					tmpColorBuffer[vertexBufferNumPoints][1] = 255;
					tmpColorBuffer[vertexBufferNumPoints][2] = 0;
				}
				else if(originalInputSparse[i].status==2)
				{
					tmpColorBuffer[vertexBufferNumPoints][0] = 0;
					tmpColorBuffer[vertexBufferNumPoints][1] = 0;
					tmpColorBuffer[vertexBufferNumPoints][2] = 255;
				}
				else if(originalInputSparse[i].status==3)
				{
					tmpColorBuffer[vertexBufferNumPoints][0] = 255;
					tmpColorBuffer[vertexBufferNumPoints][1] = 0;
					tmpColorBuffer[vertexBufferNumPoints][2] = 0;
				}
				else
				{
					tmpColorBuffer[vertexBufferNumPoints][0] = 255;
					tmpColorBuffer[vertexBufferNumPoints][1] = 255;
					tmpColorBuffer[vertexBufferNumPoints][2] = 255;
				}

			}
			else
			{
				tmpColorBuffer[vertexBufferNumPoints][0] = originalInputSparse[i].color[pnt];
				tmpColorBuffer[vertexBufferNumPoints][1] = originalInputSparse[i].color[pnt];
				tmpColorBuffer[vertexBufferNumPoints][2] = originalInputSparse[i].color[pnt];
			}
			vertexBufferNumPoints++;

			if(originalInputSparse[i].status==0||originalInputSparse[i].status==1)
			{
				Vec4 lp(tmpVertexBuffer[vertexBufferNumPoints][0], tmpVertexBuffer[vertexBufferNumPoints][1], tmpVertexBuffer[vertexBufferNumPoints][2], 1);

				Vec4 wp = camToWorld.matrix() * lp;

				pcl::PointXYZ point;
				point.x = (float)wp[0]*30;
				point.y = (float)wp[1]*30;
				point.z = (float)wp[2]*30;
				// point.r = 255;
				// point.g = 255;
				// point.b = 255;
				
				if (std::isnan(point.x) || std::isnan(point.y) || std::isnan(point.z)) continue;
				cloud->points.push_back(point);
			}
		
			assert(vertexBufferNumPoints <= numSparsePoints*patternNum);
		}
	}

	if(vertexBufferNumPoints==0)
	{
		delete[] tmpColorBuffer;
		delete[] tmpVertexBuffer;
		return true;
	}

	numGLBufferGoodPoints = vertexBufferNumPoints;
	if(numGLBufferGoodPoints > numGLBufferPoints)
	{
		numGLBufferPoints = vertexBufferNumPoints*1.3;
		vertexBuffer.Reinitialise(pangolin::GlArrayBuffer, numGLBufferPoints, GL_FLOAT, 3, GL_DYNAMIC_DRAW );
		colorBuffer.Reinitialise(pangolin::GlArrayBuffer, numGLBufferPoints, GL_UNSIGNED_BYTE, 3, GL_DYNAMIC_DRAW );
	}
	vertexBuffer.Upload(tmpVertexBuffer, sizeof(float)*3*numGLBufferGoodPoints, 0);
	colorBuffer.Upload(tmpColorBuffer, sizeof(unsigned char)*3*numGLBufferGoodPoints, 0);
	bufferValid=true;
	delete[] tmpColorBuffer;
	delete[] tmpVertexBuffer;


	return true;
}


void KeyFrameDisplay::GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M)
{
	Eigen::Matrix4f twc = camToWorld.matrix().cast<float>();		

	M.m[0] = twc(0,0);
	M.m[1] = twc(1,0);
	M.m[2] = twc(2,0);
	M.m[3]  = 0.0;

	M.m[4] = twc(0,1);
	M.m[5] = twc(1,1);
	M.m[6] = twc(2,1);
	M.m[7]  = 0.0;

	M.m[8] = twc(0,2);
	M.m[9] = twc(1,2);
	M.m[10] = twc(2,2);
	M.m[11]  = 0.0;

	M.m[12] = twc(0,3);
	M.m[13] = twc(1,3);
	M.m[14] = twc(2,3);
	M.m[15]  = 1.0;

}



void KeyFrameDisplay::drawCam(float lineWidth, float* color, float sizeFactor)
{
	if(width == 0)
		return;

	float sz=sizeFactor;

	glPushMatrix();

		Sophus::Matrix4f m = camToWorld.matrix().cast<float>();
		glMultMatrixf((GLfloat*)m.data());

		if(color == 0)
		{
			glColor3f(1,0,0);
		}
		else
			glColor3f(color[0],color[1],color[2]);

		glLineWidth(lineWidth);
		glBegin(GL_LINES);
		glVertex3f(0,0,0);
		glVertex3f(sz*(0-cx)/fx,sz*(0-cy)/fy,sz);
		glVertex3f(0,0,0);
		glVertex3f(sz*(0-cx)/fx,sz*(height-1-cy)/fy,sz);
		glVertex3f(0,0,0);
		glVertex3f(sz*(width-1-cx)/fx,sz*(height-1-cy)/fy,sz);
		glVertex3f(0,0,0);
		glVertex3f(sz*(width-1-cx)/fx,sz*(0-cy)/fy,sz);

		glVertex3f(sz*(width-1-cx)/fx,sz*(0-cy)/fy,sz);
		glVertex3f(sz*(width-1-cx)/fx,sz*(height-1-cy)/fy,sz);

		glVertex3f(sz*(width-1-cx)/fx,sz*(height-1-cy)/fy,sz);
		glVertex3f(sz*(0-cx)/fx,sz*(height-1-cy)/fy,sz);

		glVertex3f(sz*(0-cx)/fx,sz*(height-1-cy)/fy,sz);
		glVertex3f(sz*(0-cx)/fx,sz*(0-cy)/fy,sz);

		glVertex3f(sz*(0-cx)/fx,sz*(0-cy)/fy,sz);
		glVertex3f(sz*(width-1-cx)/fx,sz*(0-cy)/fy,sz);

		glEnd();
	glPopMatrix();
}


void KeyFrameDisplay::drawPC()
{

	if(!dbufferValid || dnumGLBufferGoodPoints==0)
		return;


	glDisable(GL_LIGHTING);

	glPushMatrix();

		Sophus::Matrix4f m = camToWorld.matrix().cast<float>();
		glMultMatrixf((GLfloat*)m.data());

		glPointSize(1);


		dcolorBuffer.Bind();
		glColorPointer(dcolorBuffer.count_per_element, dcolorBuffer.datatype, 0, 0);
		glEnableClientState(GL_COLOR_ARRAY);

		dvertexBuffer.Bind();
		glVertexPointer(dvertexBuffer.count_per_element, dvertexBuffer.datatype, 0, 0);
		glEnableClientState(GL_VERTEX_ARRAY);
		glDrawArrays(GL_POINTS, 0, dnumGLBufferGoodPoints);
		glDisableClientState(GL_VERTEX_ARRAY);
		dvertexBuffer.Unbind();

		glDisableClientState(GL_COLOR_ARRAY);
		dcolorBuffer.Unbind();

	glPopMatrix();
}

void KeyFrameDisplay::drawPC(float pointSize)
{

	if(!bufferValid || numGLBufferGoodPoints==0)
		return;


	glDisable(GL_LIGHTING);

	glPushMatrix();

		Sophus::Matrix4f m = camToWorld.matrix().cast<float>();
		glMultMatrixf((GLfloat*)m.data());

		glPointSize(pointSize);


		colorBuffer.Bind();
		glColorPointer(colorBuffer.count_per_element, colorBuffer.datatype, 0, 0);
		glEnableClientState(GL_COLOR_ARRAY);

		vertexBuffer.Bind();
		glVertexPointer(vertexBuffer.count_per_element, vertexBuffer.datatype, 0, 0);
		glEnableClientState(GL_VERTEX_ARRAY);
		glDrawArrays(GL_POINTS, 0, numGLBufferGoodPoints);
		glDisableClientState(GL_VERTEX_ARRAY);
		vertexBuffer.Unbind();

		glDisableClientState(GL_COLOR_ARRAY);
		colorBuffer.Unbind();

	glPopMatrix();
}

}
}
