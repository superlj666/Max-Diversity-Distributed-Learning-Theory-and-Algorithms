#ifndef KENEL_RIDGEREGRESSION_H_
#define KENEL_RIDGEREGRESSION_H_
#include "util.h"
#include "ps/ps.h"

namespace rr {
class KernelRidgeRegression {
public:
  explicit KernelRidgeRegression(Dataset& dataset, float lambda=0.1, float gamma=0.01, float sigma_=2);
  virtual ~KernelRidgeRegression() {
    delete w0_;
    delete wR_;
    delete w_;
  }

  void SetwR_(vector<float> weights);

  float* Getw();

private:
  bool GetW0();

  Dataset& dataset_;
  float lambda_;
  float gamma_;  
  float sigma_;

  float* w0_;
  float* wR_;
  float* w_;
};

}  // namespace distlr

#endif  // KENEL_RIDGEREGRESSION_H_
