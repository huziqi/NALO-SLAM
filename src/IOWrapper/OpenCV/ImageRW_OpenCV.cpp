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



#include "IOWrapper/ImageRW.h"
#include <opencv2/highgui/highgui.hpp>
#include "IOWrapper/ImageDisplay.h"
#include <opencv2/opencv.hpp>


namespace dso
{

namespace IOWrap
{
MinimalImageB* readImageBW_8U(std::string filename)
{
	cv::Mat m = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
	if(m.rows*m.cols==0)
	{
		printf("cv::imread could not read image %s! this may segfault. \n", filename.c_str());
		return 0;
	}
	if(m.type() != CV_8U)
	{
		printf("cv::imread did something strange! this may segfault. \n");
		return 0;
	}
	MinimalImageB* img = new MinimalImageB(m.cols, m.rows);
	memcpy(img->data, m.data, m.rows*m.cols);
	return img;
}

float* resizeMask(float * result, unsigned char* image_in, int wOrg, int hOrg, int w, int h)
{
	cv::Mat mask_(hOrg,wOrg,CV_8U);
	memcpy(mask_.data, image_in, hOrg * wOrg);

	cv::Mat dst_mask;
	cv::resize(mask_, dst_mask, cv::Size(w, h),0,0,cv::INTER_NEAREST);

	unsigned char* ptr_mask=new uchar[w*h];
	memcpy(ptr_mask, dst_mask.data, w * h);

	float factor=1.0;
	for(int i=0;i<w*h;i++)
	{
		result[i] = ptr_mask[i] * factor;
	}
	delete[] ptr_mask;

	return 0;
}

void resizeColor(Vec3b* color, Vec3b* image_in, int wOrg, int hOrg, int w, int h)
{
	cv::Mat color_(hOrg,wOrg,CV_8UC3);
	memcpy(color_.data, image_in, 3*hOrg * wOrg);

	cv::Mat dst_color;
	cv::resize(color_, dst_color, cv::Size(w, h),0,0,cv::INTER_NEAREST);

	memcpy(color, dst_color.data, 3*w * h);
}

// load masks
MaskImg* readMask_8U(std::string filename)
{
	cv::Mat m = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
	if(m.rows*m.cols==0)
	{
		printf("cv::imread could not read image %s! this may segfault. \n", filename.c_str());
		return 0;
	}
	if(m.type() != CV_8U)
	{
		printf("Mask image fomat is not CV_8U! this may segfault. \n");
		return 0;
	}

	// float *img_mask = new float[m.cols * m.rows];
	// for (int i = 0; i < m.cols * m.rows; i++)
	// 	img_mask[i] = (float)m.data[i];


	MaskImg *maskimg_ = new MaskImg(m.cols, m.rows);
	maskimg_->w = m.cols;
	maskimg_->h = m.rows;

	memcpy(maskimg_->data, m.data, m.cols * m.rows);

	// for(int i=0;i<m.cols*m.rows;i++)
	// {
	// 	maskimg_->data[i] = (float)m.data[i];
	// }
	// MinimalImageB3 mas(m.cols, m.rows);
	// for(int i=0;i<m.cols*m.rows;i++)
	// {
	// 	float c = maskimg_->data[i];
	// 	if (c > 255)
	// 		c = 255;
	// 	mas.at(i) = Vec3b(c,c,c);
	// }
	// IOWrap::displayImage("Mask_immu", &mas);

	return maskimg_;
}

MinimalImageB3* readImageRGB_8U(std::string filename)
{
	cv::Mat m = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
	if(m.rows*m.cols==0)
	{
		printf("cv::imread could not read image %s! this may segfault. \n", filename.c_str());
		return 0;
	}
	if(m.type() != CV_8UC3)
	{
		printf("cv::imread did something strange! this may segfault. \n");
		return 0;
	}
	MinimalImageB3* img = new MinimalImageB3(m.cols, m.rows);
	memcpy(img->data, m.data, 3*m.rows*m.cols);
	return img;
}

MinimalImage<unsigned short>* readImageBW_16U(std::string filename)
{
	cv::Mat m = cv::imread(filename, CV_LOAD_IMAGE_UNCHANGED);
	if(m.rows*m.cols==0)
	{
		printf("cv::imread could not read image %s! this may segfault. \n", filename.c_str());
		return 0;
	}
	if(m.type() != CV_16U)
	{
		printf("readImageBW_16U called on image that is not a 16bit grayscale image. this may segfault. \n");
		return 0;
	}
	MinimalImage<unsigned short>* img = new MinimalImage<unsigned short>(m.cols, m.rows);
	memcpy(img->data, m.data, 2*m.rows*m.cols);
	return img;
}

MinimalImageB* readStreamBW_8U(char* data, int numBytes)
{
	cv::Mat m = cv::imdecode(cv::Mat(numBytes,1,CV_8U, data), CV_LOAD_IMAGE_GRAYSCALE);
	if(m.rows*m.cols==0)
	{
		printf("cv::imdecode could not read stream (%d bytes)! this may segfault. \n", numBytes);
		return 0;
	}
	if(m.type() != CV_8U)
	{
		printf("cv::imdecode did something strange! this may segfault. \n");
		return 0;
	}
	MinimalImageB* img = new MinimalImageB(m.cols, m.rows);
	memcpy(img->data, m.data, m.rows*m.cols);
	return img;
}



void writeImage(std::string filename, MinimalImageB* img)
{
	cv::imwrite(filename, cv::Mat(img->h, img->w, CV_8U, img->data));
}
void writeImage(std::string filename, MinimalImageB3* img)
{
	cv::imwrite(filename, cv::Mat(img->h, img->w, CV_8UC3, img->data));
}
void writeImage(std::string filename, MinimalImageF* img)
{
	cv::imwrite(filename, cv::Mat(img->h, img->w, CV_32F, img->data));
}
void writeImage(std::string filename, MinimalImageF3* img)
{
	cv::imwrite(filename, cv::Mat(img->h, img->w, CV_32FC3, img->data));
}

}

}
