#include "ps/ps.h"
#include "kernel_ridge_regression.h"
#include "cmath"
#include <ctime>
#include <iostream>
#include <iomanip>
#include <fstream>

namespace rr
{

KernelRidgeRegression::KernelRidgeRegression(Dataset &dataset, float lambda, float gamma, float sigma)
    : dataset_(dataset), lambda_(lambda), gamma_(gamma), sigma_(sigma)
{
  w0_ = new float[dataset.n]();
  wR_ = new float[dataset.n]();
  w_ = new float[dataset.n]();
  GetW0();
}

void KernelRidgeRegression::SetwR_(vector<float> weights)
{
  for (auto i = 0; i < weights.size(); ++i)
  {
    wR_[i] = weights[i];
  }
}

bool KernelRidgeRegression::GetW0()
{
  float *X = dataset_.feature;
  float *y = dataset_.label;
  int n = dataset_.n;
  int d = dataset_.d;

  float *K = new float[n * n]();
  if (K == NULL)
  {
    delete K;
    return false;
  }

  // Kernel Matrix
  rr::GaussianKernel(dataset_, K, sigma_);

  // Part A
  for (int i = 0; i < n; ++i)
  {
    K[i * n + i] += lambda_;
  }

  // Inversion of Part A
  if (!MatrixInversion(K, n))
  {
    delete K;
    return false;
  }

  // y is the Part b
  // calculate w0_
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              n, 1, n, 1/(float)n, K, n, y, 1, 0, w0_, 1);

  delete K;
  return true;
}

float *KernelRidgeRegression::Getw()
{
  float tmp_const = 0;
  for (int i = 0; i < dataset_.n; ++i)
  {
    tmp_const += wR_[i] * w0_[i];
  }
  for (int i = 0; i < dataset_.n; ++i)
  {
    w_[i] = w0_[i] - dataset_.n * gamma_ * tmp_const / dataset_.label[i];
  }
  return w_;
}
} // namespace krr
// namespace distlr
