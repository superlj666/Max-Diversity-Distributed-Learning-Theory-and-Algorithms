#include "ps/ps.h"
#include "dist_krr.h"
#include "cmath"
#include <ctime>
#include <iostream>
#include <iomanip>
#include <fstream>

namespace rr
{

DistKRR::DistKRR(KernelData &selfKernel, float* y, float lambda, float gamma, float sigma)
    : selfKernel_(selfKernel), y_(y), lambda_(lambda), gamma_(gamma), sigma_(sigma)
{
  cout << "construct distKRR" << endl;
  w0_ = new float[selfKernel.row_size]();
  wR_ = new float[selfKernel.row_size]();
  w_ = new float[selfKernel.row_size]();
  GetW0();
  cout << "finishe construction" << endl;
}

void DistKRR::SetwR_(vector<float> weights)
{
  for (auto i = 0; i < weights.size(); ++i)
  {
    wR_[i] = weights[i];
  }
}

bool DistKRR::GetW0()
{
  int n = selfKernel_.row_size;

  // Part A
  for (int i = 0; i < n; ++i)
  {
    selfKernel_.kernel[i * n + i] += lambda_ ;
  }

  // Inversion of Part A
  if (!MatrixInversion(selfKernel_.kernel, n))
  {
    cout << "is not a Full Rank Matrix" << endl;
    return false;
  }  

  // y is the Part b
  // calculate w0_
  cout << "start computing w0..." << endl;
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              n, 1, n, 1, selfKernel_.kernel, n, y_, 1, 0, w0_, 1);

  cout << "finish computing w0..." << endl;

  cout << "w0: " << endl;
  rr::PrintMatrix(n, n, w0_);

  return true;
}

float *DistKRR::Getw()
{
  float tmp_const = 0;
  for (int i = 0; i < selfKernel_.row_size; ++i)
  {
    tmp_const += wR_[i] * w0_[i];
  }
  for (int i = 0; i < selfKernel_.row_size; ++i)
  {
    w_[i] = w0_[i] - selfKernel_.row_size * gamma_ * tmp_const / y_[i];
  }
  return w_;
}
} // namespace krr
// namespace distlr
