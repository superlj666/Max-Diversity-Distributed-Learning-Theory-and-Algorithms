#include "ps/ps.h"
#include "ridge_regression/"
#include "ridge_regression/ridge_regression.h"
#include "cmath"
#include "mkl.h"
#include <ctime>
#include <iostream>
#include <iomanip>
#include <fstream>

namespace rr {

RidgeRegression::RidgeRegression(Dataset& dataset, float lambda, float gamma, ps::KVWorker<float>* kv)
   : dataset_(dataset), lambda_(lambda), gamma_(gamma), kv_(kv) {  
  c0_=(float*)mkl_malloc(dataset.d*1*sizeof(float),64);   
  b_=(float*)mkl_malloc(dataset.d*1*sizeof(float),64);
  cR_=(float*)mkl_malloc(dataset.d*1*sizeof(float),64);
  w_=(float*)mkl_malloc(dataset.d*1*sizeof(float),64);
  GetC0();
}

float* RidgeRegression::GetCR_(){
  return cR_;
}

bool RidgeRegression::GetC0() {
  float* X=dataset_.feature;
  float* y=dataset_.label;
  int n=dataset_.n;
  int d=dataset_.d;
  
  float* A=(float*)mkl_malloc(d*d*sizeof(float),64);
  if (A==NULL) {
    mkl_free(A);
    return false;
  }    
  GetPartA(X, n, d, lambda_-gamma_, A);
  if(!MatrixInversion(A,d)) {
    mkl_free(A);
    return false;
  }
  cout << "Print inversion matrix of A:" << endl;
  PrintMatrix(d, d, A);

  GetPartb(X, y, n, d, b_);
  cout << "Print vector of b:" << endl;
  PrintMatrix(d, 1, b_);
  
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
          d, 1, d, 1, A, d, b_, 1, 0, c0_, 1);
  cout << "Print vector of c0:" << endl;
  PrintMatrix(d, 1, c0_);

  mkl_free(A);
  return true;
}

float* RidgeRegression::Getw(){
  float tmp_const=0;
  for (int i=0; i<dataset_.d; ++i) {
    tmp_const += cR_[i]*c0_[i];
  }
  for (int i=0; i<dataset_.d; ++i) {
    w_[i]=c0_[i]-gamma_*tmp_const/b_[i];
  }
  cout << "Print vector of w:" << endl;
  PrintMatrix(dataset_.d, 1, w_);
  return w_;
}


void RidgeRegression::Predict(Dataset& test, float* result) {
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
          test.n, 1, test.d, 1, test.feature, test.d, w_, 1, 0, result, 1);  
  cout << "Print vector of predict:" << endl;
  PrintMatrix(test.n, 1, result);
}

float RidgeRegression::MSE(Dataset& test){
  float* predict=(float*)mkl_malloc(test.d*1*sizeof(float),64);
  Predict(test, predict);
  float mse=0;
  for (int i=0; i<test.n; ++i) {
    mse += pow(predict-test.label,2);
  }
  mkl_free(predict);
  return mse/test.n;
}
}
// namespace distlr
