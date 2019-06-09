#include "ps/ps.h"
#include "ridge_regression.h"
#include "cmath"
#include <ctime>
#include <iostream>
#include <iomanip>
#include <fstream>

namespace rr
{

RidgeRegression::RidgeRegression(Dataset &dataset, float lambda, float gamma)
    : dataset_(dataset), lambda_(lambda), gamma_(gamma)
{
  w0_ = new float[dataset.d]();
  b_ = new float[dataset.d]();
  wR_ = new float[dataset.d]();
  w_ = new float[dataset.d]();
  A_ = new float[dataset.d * dataset.d]();
  GetW0();
}

void RidgeRegression::SetwR_(vector<float> weights)
{
  for (auto i = 0; i < weights.size(); ++i)
  {
    wR_[i] = weights[i];
  }
}

bool RidgeRegression::GetW0()
{
  float *X = dataset_.feature;
  float *y = dataset_.label;
  int n = dataset_.n;
  int d = dataset_.d;

  GetPartA(X, n, d, lambda_, A_);

  if (!MatrixInversion(A_, d))
  {
    return false;
  }

  GetPartb(X, y, n, d, b_);

  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              d, 1, d, 1, A_, d, b_, 1, 0, w0_, 1);

  cout << "Print inversion matrix of A_:" << endl;
  PrintMatrix(d, d, A_);

  cout << "Print vector of b:" << endl;
  PrintMatrix(d, 1, b_);

  cout << "Print vector of w0:" << endl;
  PrintMatrix(d, 1, w0_);

  return true;
}

float *RidgeRegression::Getw()
{
  // float tmp_const = 0;
  // for (int i = 0; i < dataset_.d; ++i)
  // {
  //   tmp_const += wR_[i] * w0_[i];
  // }
  // for (int i = 0; i < dataset_.d; ++i)
  // {
  //   w_[i] = w0_[i] - gamma_ * tmp_const / (b_[i] + 0.000000001);
  // }

  for (int i = 0; i < dataset_.d; ++i)
  {
    w_[i] = w0_[i];
  }
  cout << "Print matrix of invA to set w:" << endl;
  PrintMatrix(dataset_.d, dataset_.d, A_);

  cout << "Print vector of w0:" << endl;
  PrintMatrix(dataset_.d, 1, w_);
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              dataset_.d, 1, dataset_.d, -gamma_, A_, dataset_.d, wR_, 1, 1, w_, 1);

  cout << "Print vector of set w:" << endl;
  PrintMatrix(dataset_.d, 1, w_);
  return w_;
}
} // namespace rr
// namespace distlr
