//==============================================================
//
// SAMPLE SOURCE CODE - SUBJECT TO THE TERMS OF SAMPLE CODE LICENSE AGREEMENT,
// http://software.intel.com/en-us/articles/intel-sample-source-code-license-agreement/
//
// Copyright 2016-2017 Intel Corporation
//
// THIS FILE IS PROVIDED "AS IS" WITH NO WARRANTIES, EXPRESS OR IMPLIED, INCLUDING BUT
// NOT LIMITED TO ANY IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
// PURPOSE, NON-INFRINGEMENT OF INTELLECTUAL PROPERTY RIGHTS.
//
// =============================================================
/*******************************************************************************
*   This example measures performance of computing the real matrix product 
*   C=alpha*A*B+beta*C using Intel(R) MKL function dgemm, where A, B, and C are 
*   matrices and alpha and beta are double precision scalars. 
*
*   In this simple example, practices such as memory management, data alignment, 
*   and I/O that are necessary for good programming style and high Intel(R) MKL 
*   performance are omitted to improve readability.
********************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sstream>
#include <fstream>
#include <iostream>
#include <vector>
#include <time.h>
#include "ridge_regression.h"
using namespace std;

/* Consider adjusting LOOP_COUNT based on the performance of your computer */
/* to make sure that total run time is at least 1 second */
#define LOOP_COUNT 10
statistics int train_size_ = rr::ToInt(ps::Environment::Get()->find("TRAIN_SAMPLE_SIZE"));
int main(int argc, char *argv[])
{
     clock_t tic = clock();
    // float X[]={
    //     1,0,0,
    //     0,1,0,
    //     0,0,1,
    //     0,0,0
    // };

    // float Xp[]={
    //     1,0,0,
    //     0,1,0,
    //     0,0,1
    // };
    // float y[]={
    //     1,
    //     1,
    //     0.5,
    //     0
    // };
    // int n=4;
    // int d=3;

    // float* A=(float*)mkl_malloc(d*d*sizeof(float),64);
    // float* c0_=(float*)mkl_malloc(d*1*sizeof(float),64);
    // float* b_=(float*)mkl_malloc(d*1*sizeof(float),64);
    // float* cR_=(float*)mkl_malloc(d*1*sizeof(float),64);
    // float* w_=(float*)mkl_malloc(d*1*sizeof(float),64);
    // float lambda_=0.02;
    // float gamma_=0.01;
    // cR_[0]=0.9;
    // cR_[1]=0.1;
    // cR_[2]=0.1;

    // if (A==NULL) {
    //     mkl_free(A);
    //     return false;
    // }
    // GetPartA(X, n, d, lambda_-gamma_, A);
    // if(!MatrixInversion(A,d)) {
    //     mkl_free(A);
    //     return false;
    // }
    // cout << "Print inversion matrix of A:" << endl;
    // PrintMatrix(d, d, A);

    // GetPartb(X, y, n, d, b_);
    // cout << "Print vector of b:" << endl;
    // PrintMatrix(d, 1, b_);

    // cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    //         d, 1, d, 1, A, d, b_, 1, 0, c0_, 1);
    // cout << "Print vector of c0:" << endl;
    // PrintMatrix(d, 1, c0_);

    // float tmp_const=0;
    // for (int i=0; i<d; ++i) {
    //     tmp_const += cR_[i]*c0_[i];
    // }
    // for (int i=0; i<d; ++i) {
    //     w_[i]=c0_[i]-gamma_*tmp_const/b_[i];
    // }
    // cout << "Print vector of w:" << endl;
    // PrintMatrix(d, 1, w_);

    // float* result=(float*)mkl_malloc(3*1*sizeof(float),64);
    // cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
    //         3, 1, 3, 1, Xp, 3, w_, 1, 0, result, 1);
    // cout << "Print vector of predict:" << endl;
    // PrintMatrix(3, 1, result);

    // mkl_free(A);
    // mkl_free(c0_);
    // mkl_free(b_);
    // mkl_free(cR_);
    // mkl_free(w_);

    // // 加载数据
    // rr::Dataset dataset_(4177, 8);
    // rr::LoadData("abalone", dataset_);

    // // 计算inv(A),b
    // rr::RidgeRegression rr(dataset_, 0.02, 0.01);

    // vector<float> tmp = vector<float>(8,1);
    // rr.SetwR_(tmp);
    // // 更新w
    // string str = "tmp.model";
    // rr::SaveModel(str, rr.Getw(), 8);

    // // 文件IO测试c
    // rr::Dataset dataset_(4177, 8);
    // rr::LoadData("data/abalone", dataset_);
    // float *kernel = new float[4177 * 4177]();

    // clock_t tic = clock();
    // rr::GaussianKernel(dataset_, kernel, 2);
    // clock_t toc = clock();
    // cout << "Gaussian computation cost :" << (double)(toc - tic) / CLOCKS_PER_SEC << "second" << std::endl;

    // tic = clock();
    // rr::SaveModel("result/abalone_gaussian_2", kernel, 4177, 4177);
    // toc = clock();
    // cout << "SaveModel cost :" << (double)(toc - tic) / CLOCKS_PER_SEC << "second" << std::endl;

    // tic = clock();
    // rr::LoadModel("result/abalone_gaussian_2", kernel);
    // toc = clock();
    // cout << "LoadModel cost :" << (double)(toc - tic) / CLOCKS_PER_SEC << "second" << std::endl;

    // tic = clock();
    // rr::SaveModelToBinary("result/abalone_gaussian_2_b", kernel, 4177, 4177);
    // toc = clock();
    // cout << "SaveModelToBinary cost :" << (double)(toc - tic) / CLOCKS_PER_SEC << "second" << std::endl;

    // tic = clock();
    // rr::LoadModelToBinary("result/abalone_gaussian_2_b", kernel, 4177, 4177);
    // toc = clock();
    // cout << "LoadModelToBinary cost :" << (double)(toc - tic) / CLOCKS_PER_SEC << "second" << std::endl;

    // unsigned char* tmp= new unsigned char[rr::ToInt(argv[1])];

    // int n = 600;
    // int d = 8;
    // float lambda = 0.02;
    // float gamma = 0.01;
    // cout << "sample size: " << n << endl;

    // rr::Dataset dataset_(n, d);
    // rr::LoadData(argv[1], dataset_);

    // // 计算inv(A),b,c0
    // rr::RidgeRegression rr(dataset_, 0.02, 0.01);

    // clock_t tic = clock();
    // float *kernel = new float[20000 * 20000]();
    // rr::LoadModelToBinary("data/kernel/abalone/test_00-test_00", kernel, 20000, 20000);
    // clock_t toc = clock();
    // cout << "LoadModelToBinary cost :" << (double)(toc - tic) / CLOCKS_PER_SEC << "second" << std::endl;

   
    // if (argc < 7)
    // {
    //     cout << "Usage: left_file right_file left_size right_size feature_size sigma save_path" << endl;
    //     return -1;
    // }

    // string left_file = argv[1];
    // string right_file = argv[2];
    // int left_size = rr::ToInt(argv[3]);
    // int right_size = rr::ToInt(argv[4]);
    // int feature_size = rr::ToInt(argv[5]);
    // float sigma = rr::ToFloat(argv[6]);
    // string save_path = argv[7];

    // float *kernel = new float[left_size * right_size]();

    // // Low Dimensional
    // rr::Dataset left(left_size, feature_size);
    // rr::Dataset right(right_size, feature_size);
    // rr::LoadData(left_file, left);
    // rr::LoadData(right_file, right);
    // rr::GaussianKernel(left, right, kernel, sigma);

    // High Dimensional
    // rr::HDGaussianKernel(left_file, right_file, left_size, right_size, feature_size, kernel, sigma);
    // rr::SaveModel(save_path.c_str(), kernel, left_size, right_size);

    // string left_file = "/home/bd-dev/lijian/201804_NIPS/experiment/data/abalone/train_00";
    // string right_file = "/home/bd-dev/lijian/201804_NIPS/experiment/data/abalone/train_01";
    // int left_size = 557;
    // int right_size = 557;
    // int feature_size = 8;
    // float sigma = 2;
    // string save_path = "result/abalone_train_00-train_01";


    // // The kernel matrix must less than 2G
    // float *kernel = new float[left_size * right_size]();    
    // if (left_size * feature_size <= 512 * 1024 * 1024 && right_size * feature_size <= 512 * 1024 * 1024)
    // {
    //     rr::Dataset left(left_size, feature_size);
    //     rr::Dataset right(right_size, feature_size);
    //     rr::LoadData(left_file, left);
    //     rr::LoadData(right_file, right);

    //     rr::GaussianKernel(left, right, kernel, sigma);
    // } // High demensional problem
    // else
    // {
    //     rr::HDGaussianKernel(left_file, right_file, left_size, right_size, feature_size, kernel, sigma);
    // }

    // rr::SaveModel(save_path.c_str(), kernel, left_size, right_size);

    cout << train_size_ << endl;

    clock_t toc = clock();
    cout << "cost :" << (double)(toc - tic) / CLOCKS_PER_SEC << "second" << std::endl;

    return 0;
}
