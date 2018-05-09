#include "ps/ps.h"
#include "ridge_regression/ridge_regression.h"
#include "cmath"
#include <ctime>
#include <iostream>
#include <iomanip>
#include <fstream>

namespace rr
{

RidgeRegression::RidgeRegression(Dataset &dataset, float lambda, float gamma, ps::KVWorker<float> *kv)
    : dataset_(dataset), lambda_(lambda), gamma_(gamma), kv_(kv)
{
  c0_ = new float[dataset.d]();
  b_ = new float[dataset.d]();
  wR_ = new float[dataset.d]();
  w_ = new float[dataset.d]();
  GetC0();
}

void RidgeRegression::SetwR_(vector<float> weights)
{
  for (auto i = 0; i < weights.size(); ++i)
  {
    wR_[i] = weights[i];
  }
}

bool RidgeRegression::GetC0()
{
  float *X = dataset_.feature;
  float *y = dataset_.label;
  int n = dataset_.n;
  int d = dataset_.d;

  float *A = new float[d * d]();
  if (A == NULL)
  {
    delete A;
    return false;
  }
  GetPartA(X, n, d, lambda_ - gamma_, A);
  if (!MatrixInversion(A, d))
  {
    delete A;
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

  delete A;
  return true;
}

float *RidgeRegression::Getw()
{
  float tmp_const = 0;
  for (int i = 0; i < dataset_.d; ++i)
  {
    tmp_const += wR_[i] * c0_[i];
  }
  for (int i = 0; i < dataset_.d; ++i)
  {
    w_[i] = c0_[i] - gamma_ * tmp_const / b_[i];
  }
  cout << "Print vector of w:" << endl;
  PrintMatrix(dataset_.d, 1, w_);
  return w_;
}

void RidgeRegression::Predict(Dataset &test, float *result)
{
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              test.n, 1, test.d, 1, test.feature, test.d, w_, 1, 0, result, 1);
  cout << "Print vector of predict:" << endl;
  PrintMatrix(test.n, 1, result);
}

float RidgeRegression::MSE(Dataset &test)
{
  float *predict = new float[test.d]();
  Predict(test, predict);
  float mse = 0;
  for (int i = 0; i < test.n; ++i)
  {
    mse += pow(predict - test.label, 2);
  }
  delete predict;
  return mse / test.n;
}
} // namespace rr
// namespace distlr
