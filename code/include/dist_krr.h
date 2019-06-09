#ifndef DistKRR_H_
#define DistKRR_H_
#include "util.h"
#include "ps/ps.h"

namespace rr {
class DistKRR {
public:
  explicit DistKRR(KernelData &selfKernel, float* y, float lambda=0.1, float gamma=0.01, float sigma_=2);
  virtual ~DistKRR() {
    delete w0_;
    delete wR_;
    delete w_;
  }

  void SetwR_(float* weights);

  float* Getw();

private:
  bool GetW0();

  KernelData& selfKernel_;
  float* y_;
  float lambda_;
  float gamma_;  
  float sigma_;

  float* w0_;
  float* wR_;
  float* w_;
};

}  // namespace distlr

#endif 
