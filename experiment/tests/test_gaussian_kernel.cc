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

int main(int argc, char *argv[])
{
    if (argc < 7)
    {
        cout << "Usage: left_file right_file left_size right_size feature_size sigma save_path" << endl;
        return -1;
    }

    string left_file = argv[1];
    string right_file = argv[2];
    int left_size = rr::ToInt(argv[3]);
    int right_size = rr::ToInt(argv[4]);
    int feature_size = rr::ToInt(argv[5]);
    float sigma = rr::ToFloat(argv[6]);
    string save_path = argv[7];


    // The kernel matrix must less than 2G
    float *kernel = new float[left_size * right_size]();    
    if (left_size * feature_size <= 512 * 1024 * 1024 && right_size * feature_size <= 512 * 1024 * 1024)
    {
        rr::Dataset left(left_size, feature_size);
        rr::Dataset right(right_size, feature_size);
        rr::LoadData(left_file, left);
        rr::LoadData(right_file, right);

        rr::GaussianKernel(left, right, kernel, sigma);
    } // High demensional problem
    else
    {
        rr::HDGaussianKernel(left_file, right_file, left_size, right_size, feature_size, kernel, sigma);
    }

    rr::SaveModelToBinary(save_path.c_str(), kernel, left_size, right_size);
    delete kernel;

    return 0;
}
