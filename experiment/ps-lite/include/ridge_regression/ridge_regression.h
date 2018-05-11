#ifndef RR_RIDGEREGRESSION_H_
#define RR_RIDGEREGRESSION_H_
#include "util.h"
#include "ps/ps.h"

namespace rr {
class RidgeRegression {
public:
  explicit RidgeRegression(Dataset& dataset, float lambda=0.1, float gamma=0.01, ps::KVWorker<float>* kv=NULL);
  virtual ~RidgeRegression() {
    delete c0_;
    delete b_;
    delete wR_;
    delete w_;
    if (kv_) {
      delete kv_;
    }
  }

  void SetwR_(vector<float> weights);

  float* Getw();

  void Predict(Dataset& test, float* result);

  float MSE(Dataset& test);

private:
  bool GetC0();

  Dataset& dataset_;
  float lambda_;
  float gamma_;  
  ps::KVWorker<float>* kv_;

  float* c0_;
  float* b_;
  float* wR_;
  float* w_;
};

}  // namespace distlr

#endif  // RidgeRegression_H_
