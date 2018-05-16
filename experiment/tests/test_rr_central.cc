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

int main(int argc, char *argv[])
{
    if (argc < 6)
        return 0;
    // 加载数据
    string file_name = argv[1];
    string train_path = argv[2];
    string test_path = argv[3];
    int train_size = rr::ToInt(argv[4]);
    int test_size = rr::ToInt(argv[5]);
    int feature_size = rr::ToInt(argv[6]);
    float lambda = rr::ToFloat(argv[7]);

    rr::Dataset train(train_size, feature_size);
    rr::LoadData(train_path, train);
    rr::Dataset test(test_size, feature_size);
    rr::LoadData(test_path, test);

    // 计算inv(A),b
    rr::RidgeRegression regression(train, lambda, 0);

    // 计算w并保存
    float *central_w = regression.Getw();
    string str = "result/" + file_name + "_rr_central_w";
    rr::SaveModel(str.c_str(), central_w, train_size, 1);

    // 预测label并保存、计算MSE
    float *predict = new float[test.n]();
    rr::Predict(test, central_w, predict);
    str = "result/" + file_name + "_rr_central_predict";
    rr::SaveModel(str.c_str(), predict, test_size, 1);
    float mse = rr::MSE(test, predict);

    cout << file_name << " RR central MSE :" << mse << endl;

    return 0;
}
