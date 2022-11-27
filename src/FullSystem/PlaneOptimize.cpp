#include "FullSystem/PlaneOptimize.h"


using namespace Eigen;
using namespace std;

typedef Matrix<double, 6, 1> Vec6d;

namespace dso
{

float FullSystem::planeOptimize(CalibHessian calib, std::vector<FrameHessian*> fhs)
{
    float fx = calib.fxl(), fy = calib.fyl(), cx = calib.cxl(), cy = calib.cyl();
    Mat33 K = Mat33::Zero();
    K(0,0) = calib.fxl();
    K(1,1) = calib.fyl();
    K(0,2) = calib.cxl();
    K(1,2) = calib.cyl();
    K(2,2) = 1;
    // printf("fx,fy,cx,cy:%f,%f,%f,%f\n", fx, fy, cx, cy);
    int win_size = fhs.size();
    int reserve_points = 0;
    for(auto ph:fhs)
        reserve_points += ph->pointHessians.size();


    // ########################################
    // optimize global ground param
    // pi=[x,y,z,d]
    // ########################################
    // {
    //     ceres::Problem globalPlaneOpt;

    //     ceres::CostFunction *cost_function;
    //     ceres::LossFunction *loss_function = new ceres::HuberLoss(10);

    //     // 优化参数初始化
    //     FrameHessian *ref_ph = fhs.back();// 当前滑窗的最后一帧，作为优化地面的参考帧
    //     double *plane = new double[4];

    //     if(gplane[3]<0)
    //     {
    //         plane[0] = backup_gplane[0];
    //         plane[1] = backup_gplane[1];
    //         plane[2] = backup_gplane[2];
    //         plane[3] = backup_gplane[3];
    //     }
    //     else
    //     {
    //         plane[0] = gplane[0];
    //         plane[1] = gplane[1];
    //         plane[2] = gplane[2];
    //         plane[3] = gplane[3];
    //     }
        

    //     // printf("before:[%lf, %lf, %lf, %lf]\n", plane[0], plane[1], plane[2], plane[3]);

    //     // ceres::LocalParameterization *local_parameterization = new PlaneLocalParameterization();
    //     ceres::LocalParameterization *local_parameterization = new ceres::QuaternionParameterization();

    //     globalPlaneOpt.AddParameterBlock(plane, 4, local_parameterization);

    //     // 最后一帧的插值灰度图
    //     float *dIl = new float[wG[0] * hG[0]];
    //     for (int i = 0; i < wG[0] * hG[0]; i++)
    //     {
    //         dIl[i] = ref_ph->dI[i][0];
    //     }
    //     // ref_T是世界坐标系到ref的变换矩阵
    //     Vec6d ref_se3 = ref_ph->shell->camToWorld.inverse().log();
    //     Vector3d ref_trans = ref_ph->shell->camToWorld.inverse().translation();
    //     double ref_T[6]{
    //         ref_trans[0], ref_trans[1], ref_trans[2], ref_se3[3], ref_se3[4], ref_se3[5]};

    //     int num_ground = 0;
    //     for (int host_idx = 0; host_idx < win_size - 1; host_idx++)
    //     {
    //         SE3 h2t_T; Mat33 h2t_KRKi; Vec3 h2t_Ktran;
    //         h2t_T = ref_ph->shell->camToWorld.inverse() * fhs[host_idx]->shell->camToWorld;
    //         h2t_KRKi = K * h2t_T.rotationMatrix() * K.inverse();
    //         h2t_Ktran = K * h2t_T.translation();


    //         for (int pt_idx = 0; pt_idx < fhs[host_idx]->pointHessians.size();pt_idx++)
    //         {
    //             PointHessian *ph = fhs[host_idx]->pointHessians[pt_idx];
    //             if(!ph->onground)
    //                 continue;

    //             float pattern[patternNum]{};
    //             float calib[4]{fx, fy, cx, cy};
    //             float pt_uv[2]{ph->u, ph->v};
    //             double hit_uv[2]{10, 10};
    //             float idepth = ph->idepth_scaled;
    //             float * color = ph->color; // host帧上颜色
    //             Vec6d host_se3 = fhs[host_idx]->shell->camToWorld.log();
    //             Vector3d host_trans = fhs[host_idx]->shell->camToWorld.translation();
    //             double host_T[6]{
    //                 host_trans[0], host_trans[1], host_trans[2], host_se3[3], host_se3[4], host_se3[5]
    //             };
                
    //             if (idepth < 1e-4 || idepth > 1e3)
    //                 continue;

    //             if (!projectPoint(pt_uv[0], pt_uv[1], idepth, h2t_KRKi, h2t_Ktran, hit_uv[0], hit_uv[1]))
    //                 continue;

    //             // point从世界坐标系转为平面坐标
    //             Vector3d uv, uv_host;
    //             Vector4d uv_w;
    //             uv << pt_uv[0], pt_uv[1], 1;
    //             uv_host = K.inverse() * uv;
    //             uv_host /= idepth;
    //             uv_w << uv_host[0], uv_host[1], uv_host[2], 1;
    //             uv_w = ref_ph->shell->camToWorld.matrix() * uv_w;

    //             double a = 1 / sqrt(1 - plane[1] * plane[1]);
    //             if(std::isnan(a))
    //                 continue;
    //             double plane_coor[2]{
    //                 plane[0] * uv_w[2]*a - plane[2] * uv_w[0]*a,
    //                 a*(uv_w[1] * (1 - plane[1] * plane[1]) - plane[1] * (plane[0] * uv_w[0] + plane[2] * uv_w[2]))};

                
    //             // 添加灰度误差约束
    //             for (int idx = 0; idx < patternNum; idx++)
    //             {
    //                 pattern[idx] = color[idx];
    //             }

    //             // build target image grid2d
    //             ceres::Grid2D<float, 1> grid2d(dIl, 0, hG[0], 0, wG[0]);
    //             ceres::BiCubicInterpolator<ceres::Grid2D<float, 1>> get_pixel_val(grid2d);

    //             cost_function = GlobalPlaneError::Create(pattern, get_pixel_val, calib, hit_uv, ref_T, plane_coor);
    //             globalPlaneOpt.AddResidualBlock(cost_function, loss_function, plane);

    //             num_ground++;
    //         }   
    //     }
    
    //     // std::cout << "Solving ceres BA ... " << std::endl;
    //     ceres::Solver::Options options;
    //     options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR;
    //     options.minimizer_progress_to_stdout = disableCeresReport ? false : true;
    //     options.num_threads = 16;
    //     options.max_num_iterations = 50;
    //     ceres::Solver::Summary summary;
    //     ceres::Solve(options, &globalPlaneOpt, &summary);
    //     if(!disableCeresReport)
    //     std::cout << summary.BriefReport() << "\n";

    //     // look opt vals' changes
    //     {
    //         printf("global plane:[%lf, %lf, %lf, %lf]\n", plane[0], plane[1], plane[2], plane[3]);

    //         gplane[0] = plane[0];
    //         gplane[1] = plane[1];
    //         gplane[2] = plane[2];
    //         gplane[3] = plane[3];

    //         if(gplane[3]>0)
    //         {
    //             backup_gplane[0] = plane[0];
    //             backup_gplane[1] = plane[1];
    //             backup_gplane[2] = plane[2];
    //             backup_gplane[3] = plane[3];
    //         }
    //     }

    //     delete[] plane;
    //     delete[] dIl;
    // }




    // ########################################
    // optimize scale
    // ########################################
    {
        // 优化参数初始化
        double local_plane[4];
        double scale[win_size];
        for (int i = 0; i < win_size;i++)
            scale[i] = 1;
        double localscale = 1;
        // for (int host_idx = 1; host_idx < win_size; host_idx++)
        // {
        // 将全局平面变到host坐标系
        // host to target translation
        // Eigen::Vector4d pih, piw;
        // piw << gplane[0], gplane[1], gplane[2], gplane[3];
        // pih = fhs[win_size - 2]->PRE_camToWorld.matrix().transpose() * piw;
        // printf("transformed plane:[%lf, %lf, %lf, %lf]\n", pih[0], pih[1], pih[2], pih[3]);
        // printf("plane from image: [%lf, %lf, %lf, %lf]\n", fhs[win_size - 2]->groundP[0],
        //        fhs[win_size - 2]->groundP[1],
        //        fhs[win_size - 2]->groundP[2],
        //        fhs[win_size - 2]->groundP[3]);

        if (fhs[win_size - 1]->haveground)
        {
            local_plane[0] = fhs[win_size - 1]->groundP[0];
            local_plane[1] = fhs[win_size - 1]->groundP[1];
            local_plane[2] = fhs[win_size - 1]->groundP[2];
            local_plane[3] = fhs[win_size - 1]->groundP[3];

            // scale[win_size - 1] = pih[3] / local_plane[3];
            localscale = getlocalgh() / local_plane[3];
            // cout << "scale: " << localscale << endl;

            // compute the rotation angle in the silding window
            Vector3d rotation0= fhs[0]->shell->camToWorld.log().tail<3>();
            Vector3d rotation1= fhs.back()->shell->camToWorld.log().tail<3>();
            Vector3d angle_delta= rotation0-rotation1;
            cout<<local_plane[3]<<", "<< fhs[win_size-1]->shell->id<<", "<<angle_delta.squaredNorm()<<endl;
        }

        // printf("global:[%lf, %lf, %lf, %lf]\n", gplane_[0], gplane_[1], gplane_[2], gplane_[3]);
        // printf("plane:[%lf, %lf, %lf, %lf]\n", plane[0], plane[1], plane[2], plane[3]);
        // printf("local:[%lf, %lf, %lf, %lf]\n", local_plane[0], local_plane[1], local_plane[2], local_plane[3]);
        // }

        // Vec6d se3_log = fhs[win_size - 1]->shell->camToWorld.inverse().log();
        // Vector3d t_log = fhs[win_size - 1]->shell->camToWorld.inverse().translation();

        
        // for (int i = 0; i < win_size - 1; i++)
        // {
        //     if(scale[i]<0.1||scale[i]>3) continue;

        //     if(fhs[i+1]->scaleFixed) continue;


            // SE3 tar2host;
            // tar2host = fhs[i]->shell->camToWorld.inverse() * fhs[i + 1]->shell->camToWorld;
            // tar2host.translation() *= scale[i+1];
            // fhs[i + 1]->shell->camToWorld = fhs[i]->shell->camToWorld * tar2host;

            // for (int pt_idx = 0; pt_idx < fhs[i+1]->pointHessians.size();pt_idx++)
            // {
            //     double idepth_position = fhs[i + 1]->pointHessians[pt_idx]->idepth_scaled * scale[i+1];
            //     fhs[i+1]->pointHessians[pt_idx]->setIdepth(idepth_position);
            //     fhs[i+1]->pointHessians[pt_idx]->setIdepthZero(idepth_position);
            // }SO3d R = SO3d::exp(Vector3d(opt_vals[3],opt_vals[4],opt_vals[5]));

            // fhs[i + 1]->scaleFixed = true;
        // }
        // cout << "scale: " << scale[win_size-1] << endl;


        // printf("origin newest frame T:[%lf, %lf, %lf, %lf, %lf, %lf]\n",
        //        t_log[0], t_log[1], t_log[2], se3_log[3], se3_log[4], se3_log[5]);



        //修改倒数第一帧的尺度
        FrameHessian *fs = fhs[win_size - 1];
        if (!fs->scaleFixed)
        {
            SE3 cam2ref;
            cam2ref = fs->shell->camToTrackingRef;
            cam2ref.translation() *= localscale;

            // Matrix3d rs = fs->PRE_camToWorld.rotationMatrix();
            // rs = rs * localscale;
            // fs->PRE_camToWorld.setRotationMatrix(rs);

            fs->PRE_camToWorld = fs->shell->trackingRef->camToWorld * cam2ref;
            fs->PRE_worldToCam = fs->PRE_camToWorld.inverse();

            for (int pt_idx = 0; pt_idx < fs->pointHessians.size();pt_idx++)
            {
                double idepth_position = fs->pointHessians[pt_idx]->idepth_scaled / localscale;
                fs->pointHessians[pt_idx]->setIdepth(idepth_position);
                fs->pointHessians[pt_idx]->setIdepthZero(idepth_position);
            }

            // 位最新关键帧设置线性点
            Vec10 newStateZero = Vec10::Zero();
            newStateZero.segment<2>(6) = fs->get_state().segment<2>(6);

            fs->setEvalPT(fs->PRE_worldToCam, newStateZero);
            ef->setAdjointsF(&Hcalib);

            // 设置帧间预计算位姿变换
            fs->targetPrecalc.resize(frameHessians.size());
            for(unsigned int i=0;i<frameHessians.size();i++)
                fs->targetPrecalc[i].set(fs, frameHessians[i], &Hcalib);

            // 固定新的残差，删除较大残差
            // linearizeAll(true);

            // 更新最新帧的shell位姿
            fs->shell->camToWorld = fs->PRE_camToWorld;

            fs->scaleFixed = true;
        }
    }

    return 0;
}

// 雅各比手动添加
float FullSystem::SWGrayOptimize_J(std::vector<FrameHessian*> fhs)
{
    ceres::Problem problem;
    
    ceres::CostFunction *cost_function;
    ceres::LossFunction *loss_function = new ceres::HuberLoss(100);
   

    // 优化参数初始化
    int win_size = fhs.size();
    double *opt_T = new double[win_size * 6];
    size_t pt_nums[win_size];

    pt_nums[0] = 0;
    for (int fhs_idx = 0; fhs_idx < win_size; fhs_idx++)
    {
        double *tc2w = opt_T + fhs_idx * 6;
        Vec6d se3 = fhs[fhs_idx]->shell->camToWorld.inverse().log();

        tc2w[0] = fhs[fhs_idx]->shell->camToWorld.inverse().translation().x();
        tc2w[1] = fhs[fhs_idx]->shell->camToWorld.inverse().translation().y();
        tc2w[2] = fhs[fhs_idx]->shell->camToWorld.inverse().translation().z();   
        tc2w[3] = se3[3];
        tc2w[4] = se3[4];
        tc2w[5] = se3[5];

        if(fhs_idx<win_size-1)
            pt_nums[fhs_idx+1] = fhs[fhs_idx]->pointHessians.size() + pt_nums[fhs_idx];

        problem.AddParameterBlock(opt_T + fhs_idx * 6, 6);

        // printf("before_frame%d T:[%lf, %lf, %lf, %lf, %lf, %lf]\n", fhs_idx,
            //    tc2w[0], tc2w[1], tc2w[2],tc2w[3], tc2w[4], tc2w[5]);
    }
    double *record_f = opt_T + (win_size - 1) * 6;
    // printf("old newest frame T:[%lf, %lf, %lf, %lf, %lf, %lf]\n",
    //        record_f[0], record_f[1], record_f[2], record_f[3], record_f[4], record_f[5]);

    int reserve_points = 0;
    for(auto ph:fhs)
        reserve_points += ph->pointHessians.size();
    double *idepths = new double[reserve_points];
    for (int id = 0; id < win_size;id++)
    {
        for (int phs_idx = 0; phs_idx<fhs[id]->pointHessians.size();phs_idx++)
        {
            idepths[pt_nums[id] + phs_idx] = fhs[id]->pointHessians[phs_idx]->idepth_scaled;
            problem.AddParameterBlock(idepths + pt_nums[id] + phs_idx, 1);
        }
    }

    // 滑窗前k-1个帧上的点投影到最新帧上，优化最新帧的位姿
    for (int host_idx = 0; host_idx < win_size; host_idx++)
    {
        // printf("host frame%d has %d points\n", host_idx, fhs[host_idx]->pointHessians.size());
        for (int target_idx = 0; target_idx < win_size; target_idx++)
            // int target_idx = win_size - 1;
            {
                if (host_idx != target_idx)
                {
                    double *host_T = opt_T + host_idx * 6;
                    double *target_T = opt_T + target_idx * 6;

                    for (int pt_idx = 0; pt_idx < fhs[host_idx]->pointHessians.size();pt_idx++)
                    {
                        PointHessian *ph = fhs[host_idx]->pointHessians[pt_idx];

                        double *idepth_position = idepths + pt_idx+ pt_nums[host_idx];

                        if (idepth_position[0] < 1e-4 || idepth_position[0] > 1e3)
                            continue;
                
                        GrayTHFactor_TH *Gth = new GrayTHFactor_TH(fhs[host_idx], fhs[target_idx], ph, &Hcalib);
                        problem.AddResidualBlock(Gth, loss_function, host_T, target_T, idepth_position);

                    }
                }
            }
    }
    // std::cout << "Solving ceres BA ... " << std::endl;
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = disableCeresReport ? false : true;
    options.num_threads = 16;
    options.max_num_iterations = 50;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    if(!disableCeresReport)
    std::cout << summary.BriefReport() << "\n";

    // look opt vals' changes
    {
        // boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
        // for (int i = win_size - 4; i < win_size;i++)
        // {
            // 只修改最新关键帧的位姿
            double *opt_vals = opt_T + (win_size-1) * 6;
            // printf("opted newestframe T:[%lf, %lf, %lf, %lf, %lf, %lf]\n",
            //         opt_vals[0], opt_vals[1], opt_vals[2],
            //         opt_vals[3], opt_vals[4], opt_vals[5]);

            Eigen::Vector3d t(opt_vals[0],opt_vals[1],opt_vals[2]);
            SO3d R = SO3d::exp(Vector3d(opt_vals[3],opt_vals[4],opt_vals[5]));
            // 更新预计算量
            fhs.back()->PRE_worldToCam.setRotationMatrix(R.matrix());
            fhs.back()->PRE_worldToCam.translation() = t;
            fhs.back()->PRE_camToWorld=fhs.back()->PRE_worldToCam.inverse();

            // 更新除最新帧外滑窗内其他帧的逆深度
            for (int i = 0; i < win_size - 2;i++)
                for (int pt_idx = 0; pt_idx < fhs[i]->pointHessians.size(); pt_idx++)
                {
                    double *idepth_position = idepths + pt_idx + pt_nums[i];
                    fhs[i]->pointHessians[pt_idx]->setIdepth(*idepth_position);
                    fhs[i]->pointHessians[pt_idx]->setIdepthZero(*idepth_position);
                }

            // 位最新关键帧设置线性点
            Vec10 newStateZero = Vec10::Zero();
            newStateZero.segment<2>(6) = frameHessians.back()->get_state().segment<2>(6);

            frameHessians.back()->setEvalPT(frameHessians.back()->PRE_worldToCam, newStateZero);
            ef->setAdjointsF(&Hcalib);
            
            // 设置帧间预计算位姿变换
            for(FrameHessian* fh : frameHessians)
            {
                fh->targetPrecalc.resize(frameHessians.size());
                for(unsigned int i=0;i<frameHessians.size();i++)
                    fh->targetPrecalc[i].set(fh, frameHessians[i], &Hcalib);
            }

            // 固定新的残差，删除较大残差
            // linearizeAll(true);

            // 更新最新帧的shell位姿
            fhs.back()->shell->camToWorld = fhs.back()->PRE_camToWorld;

            // }
    }
       

    delete[] idepths;
    delete[] opt_T;
    // delete[] Pidepths;

    return 0;
}

float FullSystem::SWGrayOptimize(CalibHessian calib, std::vector<FrameHessian*> fhs)
{
    ceres::Problem problem;

    float fx = calib.fxl(), fy = calib.fyl(), cx = calib.cxl(), cy = calib.cyl();
    Mat33 K = Mat33::Zero();
    K(0,0) = calib.fxl();
    K(1,1) = calib.fyl();
    K(0,2) = calib.cxl();
    K(1,2) = calib.cyl();
    K(2,2) = 1;
    // printf("fx,fy,cx,cy:%f,%f,%f,%f\n", fx, fy, cx, cy);


    int reserve_points = 0;
    for(auto ph:fhs)
        reserve_points += ph->pointHessians.size();
    double *idepths = new double[reserve_points];
    // double *Pidepths = new double[reserve_points];


    ceres::CostFunction *cost_function;
    ceres::LossFunction *loss_function = new ceres::HuberLoss(100);
   
    Mat33 h2t_R, h2t_KRKi; Mat44 h2t_T, hRt, tRt_inv; Vec3 h2t_tran, h2t_Ktran;

    // 优化参数初始化
    int win_size = fhs.size();
    double *opt_T = new double[win_size * 6];
    size_t pt_nums[win_size];

    pt_nums[0] = 0;
    for (int fhs_idx = 0; fhs_idx < win_size; fhs_idx++)
    {
        double *tc2w = opt_T + fhs_idx * 6;
        Vec6d se3 = fhs[fhs_idx]->shell->camToWorld.log();

        tc2w[0] = fhs[fhs_idx]->shell->camToWorld.translation().x();
        tc2w[1] = fhs[fhs_idx]->shell->camToWorld.translation().y();
        tc2w[2] = fhs[fhs_idx]->shell->camToWorld.translation().z();   
        tc2w[3] = se3[3];
        tc2w[4] = se3[4];
        tc2w[5] = se3[5];

        if(fhs_idx<win_size-1)
            pt_nums[fhs_idx+1] = fhs[fhs_idx]->pointHessians.size() + pt_nums[fhs_idx];

        problem.AddParameterBlock(opt_T + fhs_idx * 6, 6);

        printf("before_frame%d T:[%lf, %lf, %lf, %lf, %lf, %lf]\n", fhs_idx,
               tc2w[0], tc2w[1], tc2w[2],tc2w[3], tc2w[4], tc2w[5]);
    }

    for (int id = 0; id < win_size;id++)
    {
        for (int phs_idx = 0; phs_idx<fhs[id]->pointHessians.size();phs_idx++)
        {
            idepths[pt_nums[id] + phs_idx] = fhs[id]->pointHessians[phs_idx]->idepth_scaled;
            problem.AddParameterBlock(idepths + pt_nums[id] + phs_idx, 1);
        }
    }

    // 添加残差
    // if(win_size<4)
    //     return 0;

    for (int host_idx = 0; host_idx < win_size; host_idx++)
    {
        // printf("host frame%d has %d points\n", host_idx, fhs[host_idx]->pointHessians.size());
        for (int target_idx = 0; target_idx < win_size; target_idx++)
        {
            if (host_idx != target_idx)
            {
                //  非当前关键帧上
                float *dIl = new float[wG[0] * hG[0]];
                for (int i = 0; i < wG[0] * hG[0]; i++)
                {
                    dIl[i] = fhs[target_idx]->dI[i][0];
                }

                // host to target translation
                hRt.block<3, 4>(0, 0) = fhs[host_idx]->shell->camToWorld.matrix3x4().block<3, 4>(0, 0);
                hRt.block<1, 4>(3, 0) = Eigen::Array4d::Constant(0);
                hRt(3, 3) = 1;
                tRt_inv.block<3, 4>(0, 0) = fhs[target_idx]->shell->camToWorld.inverse().matrix3x4().block<3, 4>(0, 0);
                tRt_inv.block<1, 4>(3, 0) = Eigen::Array4d::Constant(0);
                tRt_inv(3, 3) = 1;

                h2t_T = tRt_inv * hRt;
                h2t_R.block<3, 3>(0, 0) = h2t_T.block<3, 3>(0, 0);
                h2t_tran.block<3, 1>(0, 0) = h2t_T.block<3, 1>(0, 3);
                h2t_KRKi = K * h2t_R * K.inverse();
                h2t_Ktran = K * h2t_tran;

                // printf("now start add residual between frame_%d and frame_%d\n", host_idx, target_idx);

                double *host_T = opt_T + host_idx * 6;
                double *target_T = opt_T + target_idx * 6;

                for (int pt_idx = 0; pt_idx < fhs[host_idx]->pointHessians.size();pt_idx++)
                {
                    PointHessian *ph = fhs[host_idx]->pointHessians[pt_idx];
                    float pattern[patternNum]{};
                    float calib[4]{fx, fy, cx, cy};
                    float pt_uv[2]{ph->u, ph->v};
                    double hit_uv[2]{10, 10};
                    if (!projectPoint(pt_uv[0], pt_uv[1], ph->idepth_scaled, h2t_KRKi, h2t_Ktran, hit_uv[0], hit_uv[1]))
                        continue;

                    double *idepth_position = idepths + pt_idx+ pt_nums[host_idx];
                    
                    float * color = ph->color; // host帧上颜色
                    if (idepth_position[0] < 1e-4 || idepth_position[0] > 1e3)
                        continue;
                    // 添加灰度误差约束
                    for (int idx = 0; idx < patternNum; idx++)
                    {
                        pattern[idx] = color[idx];
                    }

                    // build target image grid2d
                    ceres::Grid2D<float, 1> grid2d(dIl, 0, hG[0], 0, wG[0]);
                    ceres::BiCubicInterpolator<ceres::Grid2D<float, 1>> get_pixel_val(grid2d);

                    cost_function = GreyprojectError::Create(pattern, get_pixel_val, calib, pt_uv, hit_uv);
                    problem.AddResidualBlock(cost_function, loss_function, host_T, target_T, idepth_position);

                }
                delete[] dIl;
            }
        }
    }
    std::cout << "Solving ceres BA ... " << std::endl;
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = disableCeresReport ? false : true;
    options.num_threads = 16;
    options.max_num_iterations = 50;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    if(!disableCeresReport)
    std::cout << summary.BriefReport() << "\n";

    // look opt vals' changes
    {
        // boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
        for (int i = win_size - 4; i < win_size;i++)
        {
            double *opt_vals = opt_T + i * 6;
            printf("after_frame%d T:[%lf, %lf, %lf, %lf, %lf, %lf]\n", i,
                    opt_vals[0], opt_vals[1], opt_vals[2],
                    opt_vals[3], opt_vals[4], opt_vals[5]);
            
            Eigen::Matrix<double, 6, 1> temp_val;
            temp_val << opt_vals[0],opt_vals[1],opt_vals[2],opt_vals[3],opt_vals[4],opt_vals[5];
            double r[9];
            double angle[3];
            angle[0] = opt_vals[3];
            angle[1] = opt_vals[4];
            angle[2] = opt_vals[5];
            ceres::AngleAxisToRotationMatrix(angle, r);
            Eigen::Matrix<double, 3, 3> R_matrix;
            R_matrix << r[0], r[3], r[6],
                        r[1], r[4], r[7],
                        r[2], r[5], r[8];
            // update poses
            fhs[i]->shell->camToWorld.setRotationMatrix(R_matrix);
            fhs[i]->shell->camToWorld.translation() = temp_val.template head<3>();

            // update idepths
            for (int pt_idx = 0; pt_idx < fhs[i]->pointHessians.size();pt_idx++)
            {
                double *idepth_position = idepths + pt_idx+ pt_nums[i];
                fhs[i]->pointHessians[pt_idx]->setIdepth(*idepth_position);
                fhs[i]->pointHessians[pt_idx]->setIdepthZero(*idepth_position);
            }
        }
    }
       

    delete[] idepths;
    delete[] opt_T;
    // delete[] Pidepths;

    return 0;
}

float FullSystem::SWGrayOptimizeG2o(CalibHessian calib, std::vector<FrameHessian*> fhs)
{
    float fx = calib.fxl(), fy = calib.fyl(), cx = calib.cxl(), cy = calib.cyl();
    Mat33 K = Mat33::Zero();
    K(0,0) = calib.fxl();
    K(1,1) = calib.fyl();
    K(0,2) = calib.cxl();
    K(1,2) = calib.cyl();
    K(2,2) = 1;

    return 0;
}

bool PlaneLocalParameterization::Plus(const double *x, const double *delta, double *x_plus_delta) const
{
    Eigen::Map<const Eigen::Quaterniond> _q(x);

    Eigen::Quaterniond dq = Utility::deltaQ(Eigen::Map<const Eigen::Vector3d>(delta));

    Eigen::Map<Eigen::Quaterniond> q(x_plus_delta);

    q = (_q * dq).normalized();

    return true;
}
bool PlaneLocalParameterization::ComputeJacobian(const double *x, double *jacobian) const
{
    Eigen::Map<Eigen::Matrix<double, 4, 3, Eigen::RowMajor>> j(jacobian);
    j.topRows<3>().setIdentity();
    j.bottomRows<1>().setZero();

    return true;
}


}