#ifndef RR_RIDGEREGRESSION_H_
#define RR_RIDGEREGRESSION_H_
#include "util.h"
#include "ps/ps.h"

namespace rr
{
class RidgeRegression
{
public:
  explicit RidgeRegression(Dataset &dataset, float lambda = 0.1, float gamma = 0.01);
  virtual ~RidgeRegression()
  {
    delete w0_;
    delete b_;
    delete wR_;
    delete w_;
  }

  void SetwR_(vector<float> weights);

  float *Getw();

private:
  bool GetW0();

  Dataset &dataset_;
  float lambda_;
  float gamma_;

  float *w0_;
  float *b_;
  float *wR_;
  float *w_;
  float *A_;
};

} // namespace rr

#endif // RidgeRegression_H_
