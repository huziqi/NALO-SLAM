#include "FullSystem/MapPoint.h"
#include "FullSystem/CoarseTracker.h"
#include <FullSystem/ImmaturePoint.h>

using namespace std;

namespace dso
{
    MapPoint::MapPoint(int u_, int v_, FrameHessian *host_)
        : u(u_), v(v_), host(host_), idepth_min(0), idepth_max(NAN)
    {
        bgr[0] = 255;
        bgr[1] = 255;
        bgr[2] = 255;
    }
    MapPoint::~MapPoint(){};

    // 将滑窗内的有深度的成熟点投影到三维空间中,points存的世界坐标系下的坐标
    void DenseMapping::keyFrameMap(std::vector<FrameHessian *> fhs, std::vector<Vec3f> &points)
    {
        for(auto fh:fhs)
        {
            for (auto mpt:fh->pointHessians)
            {
                Vec3f cP = Ki[0] * Vec3f(mpt->u, mpt->v, 1);
                cP /= mpt->idepth_scaled;
                Vec4f mP;
                mP << cP[0], cP[1], cP[2], 1;
                Vec3f point = fh->shell->camToWorld.matrix3x4().cast<float>() * mP;
                points.push_back(point);
            }
            for(auto impt:fh->immaturePoints)
            {
                Vec3f cP = Ki[0] * Vec3f(impt->u, impt->v, 1);
                cP /= (impt->idepth_max+impt->idepth_min)*0.5f;
                Vec4f mP;
                mP << cP[0], cP[1], cP[2], 1;
                Vec3f point = fh->shell->camToWorld.matrix3x4().cast<float>() * mP;
                points.push_back(point);
            }
        }
        printf("frame%d-%d has %d size local map\n", fhs[0]->frameID, fhs.back()->frameID, points.size());
    }

    // 矫正局部稠密地图
    void DenseMapping::refineMap(std::vector<FrameHessian *> fhs, FrameHessian* dfh)
    {
        std::vector<Vec3f> sparsePoints;
        keyFrameMap(fhs, sparsePoints);

        for(auto plane:dfh->lplane)
        {
            if(plane.points.size()<10) continue;

            float minx = FLT_MAX; float maxx = FLT_MIN;
            float miny = FLT_MAX; float maxy = FLT_MIN;
            float minz = FLT_MAX; float maxz = FLT_MIN;

            Vec4f pi=Vec4f::Zero();

            for (auto pt : plane.points)
            {
                Vec3f cP = Ki[0] * Vec3f(pt[0], pt[1], 1);
                cP /= pt[2];
                Vec4 mP;
                mP << cP[0], cP[1], cP[2], 1;
                mP = dfh->shell->camToWorld.matrix() * mP;

                // printf("mp:[%f,%f,%f]\n", mP[0], mP[1], mP[2]);

                if(!std::isfinite(mP[0]) || !std::isfinite(mP[1]) || !std::isfinite(mP[2])) continue;

                if(mP[0]<minx) minx = mP[0]; if(mP[0]>maxx) maxx = mP[0];
                if(mP[1]<miny) miny = mP[1]; if(mP[1]>miny) maxy = mP[1];
                if(mP[2]<minz) minz = mP[2]; if(mP[2]>minz) maxz = mP[2];
            }

            printf("3D rect:[%f,%f,%f,%f,%f,%f]\n", minx, maxx, miny, maxy, minz, maxz);

            pcl::PointCloud<pcl::PointXYZ>::Ptr raw_wpts (new pcl::PointCloud<pcl::PointXYZ>);
            pcl::PointCloud<pcl::PointXYZ>::Ptr dst_wpts (new pcl::PointCloud<pcl::PointXYZ>);
            for(auto pt:sparsePoints)
            {
                pcl::PointXYZ point;
                point.x = pt[0];
                point.y = pt[1];
                point.z = pt[2];
                raw_wpts->points.push_back(point);
            }
            // Create the filtering object
            pcl::PassThrough<pcl::PointXYZ> pass;
            pass.setInputCloud(raw_wpts);pass.setFilterFieldName("x");pass.setFilterLimits(minx, maxx);pass.filter(*dst_wpts);
            pass.setInputCloud(dst_wpts);pass.setFilterFieldName("y");pass.setFilterLimits(miny, maxy);pass.filter(*dst_wpts);
            pass.setInputCloud(dst_wpts);pass.setFilterFieldName("z");pass.setFilterLimits(minz, maxz);pass.filter(*dst_wpts);

            printf("dst_wpts size:%d\n", dst_wpts->points.size());

            if(!fitPlane(dst_wpts, pi)) continue;
            
            Vec4f wplane = dfh->shell->camToWorld.matrix().inverse().transpose().cast<float>() * plane.plane;
            Vec4f diff_pi = pi - wplane;
            float plane_dis = diff_pi.squaredNorm();

            float sum_dis = 0;
            for(auto pt:dst_wpts->points)
            {
                Vec3f n;
                n << plane.plane[0], plane.plane[1], plane.plane[2];
                float tt = abs(plane.plane[0] * pt.x + plane.plane[0] * pt.x + plane.plane[0] * pt.x + plane.plane[3]);
                sum_dis += tt / n.squaredNorm();
            }

            printf("dso plane:[%f,%f,%f,%f], dense plane:[%f,%f,%f,%f]\n",
                   pi[0], pi[1], pi[2], pi[3], wplane[0], wplane[1], wplane[2], wplane[3]);
            printf("diff plane:%f, sum distance:%f\n", plane_dis, sum_dis/dst_wpts->points.size());
        }
    }

    // 判断是否接受这块面片
    bool DenseMapping::acceptPatch(std::vector<Vec3f> sparsePoints, FrameHessian* dfh, LocalPlane plane)
    {
        if(plane.points.size()<10) return false;

        float minx = FLT_MAX; float maxx = FLT_MIN;
        float miny = FLT_MAX; float maxy = FLT_MIN;
        float minz = FLT_MAX; float maxz = FLT_MIN;

        Vec4f pi=Vec4f::Zero();

        for (auto pt : plane.points)
        {
            Vec3f cP = Ki[0] * Vec3f(pt[0], pt[1], 1);
            cP /= pt[2];
            Vec4 mP;
            mP << cP[0], cP[1], cP[2], 1;
            mP = dfh->shell->camToWorld.matrix() * mP;

            // printf("mp:[%f,%f,%f]\n", mP[0], mP[1], mP[2]);

            if(!std::isfinite(mP[0]) || !std::isfinite(mP[1]) || !std::isfinite(mP[2])) continue;

            if(mP[0]<minx) minx = mP[0]; if(mP[0]>maxx) maxx = mP[0];
            if(mP[1]<miny) miny = mP[1]; if(mP[1]>miny) maxy = mP[1];
            if(mP[2]<minz) minz = mP[2]; if(mP[2]>minz) maxz = mP[2];
        }


        pcl::PointCloud<pcl::PointXYZ>::Ptr raw_wpts (new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr dst_wpts (new pcl::PointCloud<pcl::PointXYZ>);
        for(auto pt:sparsePoints)
        {
            pcl::PointXYZ point;
            point.x = pt[0];
            point.y = pt[1];
            point.z = pt[2];
            raw_wpts->points.push_back(point);
        }
        // Create the filtering object
        pcl::PassThrough<pcl::PointXYZ> pass;
        pass.setInputCloud(raw_wpts);pass.setFilterFieldName("x");pass.setFilterLimits(minx, maxx);pass.filter(*dst_wpts);
        pass.setInputCloud(dst_wpts);pass.setFilterFieldName("y");pass.setFilterLimits(miny, maxy);pass.filter(*dst_wpts);
        pass.setInputCloud(dst_wpts);pass.setFilterFieldName("z");pass.setFilterLimits(minz, maxz);pass.filter(*dst_wpts);


        if(!fitPlane(dst_wpts, pi)) 
            return false;
        
        Vec4f wplane = dfh->shell->camToWorld.matrix().inverse().transpose().cast<float>() * plane.plane;
        Vec4f diff_pi = pi - wplane;
        float plane_dis = diff_pi.squaredNorm();

        float sum_dis = 0;
        for(auto pt:dst_wpts->points)
        {
            Vec3f n;
            n << plane.plane[0], plane.plane[1], plane.plane[2];
            float tt = abs(plane.plane[0] * pt.x + plane.plane[0] * pt.x + plane.plane[0] * pt.x + plane.plane[3]);
            sum_dis += tt / n.squaredNorm();
        }

        // 计算3D rect中包含dso稀疏点的数量
        // 判断主方向
        printf("3D rect:[%f,%f,%f,%f,%f,%f]\n", minx, maxx, miny, maxy, minz, maxz);
        float area = sqrt((maxx - minx)*(maxx - minx) + (maxz - minz)*(maxz - minz)) * (maxy - miny);
        // int major = 0;
        // float major_val = plane.plane[0];
        // for (int i = 1; i < 3; i++)
        // {
        //     if(plane.plane[i]>major_val)
        //         major = i;
        // }
        // switch (major)
        // {
        //     case 0:
        //         minx -= maxx - minx;
        //         maxx += maxx - minx;
        //         break;
        //     case 1:
        //         miny -= maxy - miny;
        //         maxy += maxy - miny;
        //         break;
        //     case 2:
        //         minz -= maxz - minz;
        //         maxz += maxz - minz;
        //         break;
        //     default:
        //         break;
        // }

        int inner = 0;
        for(auto pt:sparsePoints)
        {
            if(pt[0]<maxx && pt[0]>minx && pt[1]<maxy && pt[1]>miny && pt[2]<maxz && pt[2]>minz)
                inner++;
        }

        
        // printf("!!!!!!!!!!!!!!area:%f\n", area);

        if ((abs(plane_dis) < 0.5 || abs(sum_dis / dst_wpts->points.size()) < 2)&&(abs(plane.plane[0])>0.8||abs(plane.plane[1])>0.8||abs(plane.plane[2])>0.8))
        {
            printf("3D rect:[%f,%f,%f,%f,%f,%f]\n", minx, maxx, miny, maxy, minz, maxz);
            printf("dst_wpts size:%d\n", dst_wpts->points.size());
            printf("dso plane:[%f,%f,%f,%f], dense plane:[%f,%f,%f,%f]\n",
                    pi[0], pi[1], pi[2], pi[3], wplane[0], wplane[1], wplane[2], wplane[3]);
            printf("diff plane:%f, sum distance:%f\n", plane_dis, sum_dis / dst_wpts->points.size());
            printf("!!!!!!!!!!!!!!!!!!!inner: %d \n", inner);
            return true;  
        }
        else
            return false;
    }

    void DenseMapping::updateMap(std::vector<FrameHessian*> fhs,FrameHessian *host)
    {
        int npoint = host->pointHessians.size();
        int mpoint = host->immaturePoints.size();
        std::shared_ptr<float> mpu(new float[npoint+mpoint]);
        std::shared_ptr<float> mpv(new float[npoint+mpoint]);
        std::shared_ptr<float> midepth(new float[npoint+mpoint]);
        vector<vector<Vec4f>> clusters;

        std::vector<Vec3f> sparsePoints;
        keyFrameMap(fhs, sparsePoints);

        for (int i = 0; i < npoint;i++)
        {
            PointHessian *pp = host->pointHessians[i];
            mpu.get()[i] = pp->u;
            mpv.get()[i] = pp->v;
            midepth.get()[i] = pp->idepth_scaled;
        }
        for (int i = npoint; i < mpoint+npoint;i++)
        {
            ImmaturePoint *pp = host->immaturePoints[i-npoint];
            mpu.get()[i] = pp->u;
            mpv.get()[i] = pp->v;
            midepth.get()[i] = (pp->idepth_max+pp->idepth_min)*0.5f;
        }   

        makeMaskDistMap(host->mask, clusters, mpu.get(), mpv.get(), midepth.get(), npoint+mpoint);

        //debug
        printf("totally cluster size:%d\n", clusters.size());
        for (int i = 0; i < clusters.size();i++)
        {
            printf("cluster%d has %d points, color is %f\n", i, clusters[i].size(), clusters[i][0][3]);
        }


        for (int i = 0; i < clusters.size();i++)
        {
            Vec3f dir_vector;
            float dis_plane;
			float score;

			dir_vector << 1, 1, 1;
			dis_plane = 1;

			if(!fitPlane(clusters[i], dir_vector, dis_plane, score))
				continue;


            // printf("plane%d:平面得分：%f，平面高度：%f\n",i, score, dis_plane);

            // 找到点集中x,y的最大最小值
			int minx = INT_MAX;
			int miny = INT_MAX;
			int maxx = INT_MIN;
			int maxy = INT_MIN;
			// for (auto clu : clusters[i])
			// {
			// 	if(clu[0]>maxx) maxx = clu[0];
			// 	if(clu[0]<minx) minx = clu[0];
			// 	if(clu[1]>maxy) maxy = clu[1];
			// 	if(clu[1]<miny) miny = clu[1];
			// }


            for (int x = 2; x < wG[0]-2;x++)
            {
                for (int y = 2; y < hG[0]-2;y++)
                {
                    if(host->mask[x+y*wG[0]]!=clusters[i][0][3]) continue;
                    if(x>maxx) maxx = x;
                    if(x<minx) minx = x;
                    if(y>maxy) maxy = y;
                    if(y<miny) miny = y;
                }
            }


            Vec4f splane, rect, temp;
            splane << dir_vector[0], dir_vector[1], dir_vector[2], dis_plane;

            LocalPlane pplane;
            pplane.plane = splane;
            // pplane.rect << minx, maxx, miny, maxy;
            pplane.rect[0] = minx;
            pplane.rect[1] = maxx;
            pplane.rect[2] = miny;
            pplane.rect[3] = maxy;

            pplane.points.assign(clusters[i].begin(), clusters[i].end());
            host->lplane.push_back(pplane);
            
            // if(acceptPatch(sparsePoints,host,pplane))
            {
                makeMap(clusters[i], host, pplane);
            }
        }
    }

    void DenseMapping::makeMap(vector<Vec4f> cluster, FrameHessian* fh, LocalPlane& plane)
    {   
        // if(cluster.size()<10)
        //     return;
        //test!!!
         MinimalImageB3 img_out(w[0],h[0]);
        for(int i=0;i<w[0]*h[0];i++)
        {
            float c = fh->dI[i][0]*0.7;
            if(c>255) c=255;
            img_out.at(i) = Vec3b(c,c,c);
        }
        

        float *mask = fh->mask;
        float *dIl = new float[wG[0] * hG[0]];
        Vec3b *cIl = fh->img_bgr;
        for (int i = 0; i < wG[0] * hG[0]; i++)
        {
            dIl[i] = fh->dI[i][0];
        }
        float pcolor = cluster[0][3];
        if(pcolor==0)
            return;
        cout << "cluster size:" << cluster.size() << ", color:" << pcolor << endl;
        printf("plane:[%f,%f,%f,%f], rect:[%d,%d,%d,%d]\n", plane.plane[0], plane.plane[1], plane.plane[2], plane.plane[3],
                                                            plane.rect[0],plane.rect[1],plane.rect[2],plane.rect[3]);

        float minx = FLT_MAX; float maxx = FLT_MIN;
        float miny = FLT_MAX; float maxy = FLT_MIN;
        float minz = FLT_MAX; float maxz = FLT_MIN;
        int mpcount = 0;
        std::vector<MapPoint*> mpcache;
        for (int i = plane.rect[2]; i < plane.rect[3]; i++)
        {
            for (int j = plane.rect[0]; j < plane.rect[1];j++)
            {
                float color = mask[j + i * wG[0]];
                if(color!=pcolor) continue;

                if(i%3==0||j%3==0)//均匀选点
                {
                    MapPoint *mpoint = new MapPoint(j, i, fh);
                    float ddepth = plane.plane[0] * (j * fxi[0] - cx[0] * fxi[0]) + plane.plane[1] * (i * fyi[0] - cy[0] * fyi[0]) + plane.plane[2];
                    if(ddepth==0) continue;

                    float depth = -plane.plane[3] / ddepth;
                    if(depth==0) continue;

                    mpoint->idepth = 1 / depth;
                    mpoint->color = dIl[j + i * wG[0]];
                    mpoint->bgr = cIl[j + i * wG[0]];

                    mpcache.push_back(mpoint);
                    img_out.at(j + i * wG[0]) = Vec3b(250,0,0);

                    Vec3f cP = Ki[0] * Vec3f(j, i, 1);
                    cP /= mpoint->idepth;
                    Vec4 mP;
                    mP << cP[0], cP[1], cP[2], 1;
                    mP = fh->shell->camToWorld.matrix() * mP;
                    if(mP[0]<minx) minx = mP[0]; if(mP[0]>maxx) maxx = mP[0];
                    if(mP[1]<miny) miny = mP[1]; if(mP[1]>miny) maxy = mP[1];
                    if(mP[2]<minz) minz = mP[2]; if(mP[2]>minz) maxz = mP[2];
                    mpcount++;
                }
            }
        }

        if(maxx-minx<30&&maxy-miny<30&&maxz-minz<30)
            fh->mapPoints.insert(fh->mapPoints.end(), mpcache.begin(), mpcache.end());
        delete[] dIl;
        // IOWrap::displayImage("planes", &img_out);
    }

    void DenseMapping::makeK(CalibHessian* HCalib)
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
     * @brief 在mask上找到像素值一样的点，统计点的个数，并排序
     * 
     * @param clusters：投影点x坐标，投影点y坐标，投影点逆深度，投影点在mask上的像素值
    */
    void DenseMapping::makeMaskDistMap(float* refmask, std::vector<std::vector<Vec4f>>& clusters, 
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
            
            point << xx, yy, midepth[i], refmask[xx + yy * w[0]];
            // printf("%f, ",refmask[xx+yy*w[0]]);

            if(xx>2 && xx<width-2 && yy>2 && yy<height-2)
                waitVec.push_back(point);
        }


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


        // 根据相同像素值的点的数量逆序排序
        sort(clusters.begin(), clusters.end(), [&](std::vector<Vec4f> i, std::vector<Vec4f> j){ return i.size() > j.size(); });

        
        return;
    }

    /**
     * @brief 根据点的逆深度，拟合点云的平面参数
     * 
     * @param clusters：投影点x坐标，投影点y坐标，投影点逆深度，投影点在mask上的像素值
     * @param dir_vector: 方向向量
     * @param dis_plane: 平面距相机坐标系原点距离
    */
    bool DenseMapping::fitPlane(std::vector<Vec4f> cluster, Vec3f& dir_vector, float& dis_plane, float &score)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

        Vec3f x_axis, z_axis;
        float mid_x = 0;
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
                printf("fitplane投影到相机坐标系，坐标值不是有限值！\n");
                continue;
            }

            point.x = mP[0];
            point.y = mP[1];
            point.z = mP[2];
            cloud->points.push_back(point);

            mid_x += point.x / size_points;
            mid_z += point.z / size_points;
        }

        //printf("cloud size:%ld\n", cloud->points.size());
        if(cloud->points.size()<10)
            return false;

        pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
        // Create the segmentation object
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        // Optional
        seg.setOptimizeCoefficients (true);
        // Mandatory
        seg.setModelType (pcl::SACMODEL_PLANE);
        seg.setMethodType (pcl::SAC_RANSAC);
        seg.setDistanceThreshold (0.01);

        seg.setInputCloud (cloud);
        seg.segment (*inliers, *coefficients);

        dir_vector[0] = coefficients->values[0];
        dir_vector[1] = coefficients->values[1];
        dir_vector[2] = coefficients->values[2];
        dis_plane = coefficients->values[3];
        // printf("plane para:%lf,%lf,%lf,%lf\n", dir_vector[0], dir_vector[1], dir_vector[2], dis_plane);

        return true;
    }

    bool DenseMapping::fitPlane(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, Vec4f& pi)
    {
        //printf("cloud size:%ld\n", cloud->points.size());
        if(cloud->points.size()<10)
            return false;

        pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
        // Create the segmentation object
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        // Optional
        seg.setOptimizeCoefficients (true);
        // Mandatory
        seg.setModelType (pcl::SACMODEL_PLANE);
        seg.setMethodType (pcl::SAC_RANSAC);
        seg.setDistanceThreshold (0.01);

        seg.setInputCloud (cloud);
        seg.segment (*inliers, *coefficients);

        pi[0] = coefficients->values[0];
        pi[1] = coefficients->values[1];
        pi[2] = coefficients->values[2];
        pi[3] = coefficients->values[3];
        // printf("plane para:%lf,%lf,%lf,%lf\n", dir_vector[0], dir_vector[1], dir_vector[2], dis_plane);

        return true;
    }
}