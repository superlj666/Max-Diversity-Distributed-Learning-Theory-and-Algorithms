#include <mkl.h>
#include <math.h>
#include "ps/ps.h"
#include "ridge_regression/ridge_regression.h"
#include<time.h>
using namespace ps;

/**
 * \brief a req_handle to process request in dc method
 */
template <typename Val>
class KVDCServer
{
public:
  KVDCServer(float zeta, size_t d) : zeta_(zeta), d_(d)
  {
    using namespace std::placeholders;
    ps_server_ = new KVServer<float>(0);
    zeta_ = zeta;
    weight_new = new float[d]();
    weight_old = new float[d]();
    round_ = 0;

    ps_server_->set_request_handle(
        std::bind(&KVDCServer::ReqHandle, this, _1, _2, _3));
    std::cout << "start round " << round_ << std::endl;
  }

  ~KVDCServer()
  {
    delete weight_new;
    delete weight_old;
    if (ps_server_)
    {
      delete ps_server_;
    }
  }

private:
  void ReqHandle(const KVMeta &req_meta,
                 const KVPairs<Val> &req_data,
                 KVServer<Val> *server)
  {
    if (req_meta.push)
    {
      CHECK_EQ(d_, req_data.vals.size());

      // Update
      req_metas.push_back(req_meta);
      if (req_datas.find(req_meta.sender) == req_datas.end())
      {
        req_datas[req_meta.sender] = std::vector<float>(d_);
        weight_r[req_meta.sender] = std::vector<float>(d_);
      }
      std::cout << "push data from worker " << req_meta.sender << std::endl;
      for (size_t i = 0; i < d_; ++i)
      {
        std::cout << req_data.vals[i] << " ";
        weight_new[i] += req_data.vals[i];
        req_datas[req_meta.sender][i] = req_data.vals[i];
      }
      std::cout << std::endl;

      // Wait all push to Update wr and response
      if (req_metas.size() == (size_t)NumWorkers())
      {
        float var = 0;
        for (size_t i = 0; i < d_; ++i)
        {
          var += pow(weight_new[i] - weight_old[i], 2);
        }
        var = sqrt(var);
        std::cout << "var: " << var << std::endl;

        for (auto &req : req_metas)
        {
          if (var <= zeta_/d_)
          {
            weight_r.clear();
          }
          else
          {
            for (size_t i = 0; i < d_; ++i)
            {
              weight_r[req.sender][i] = (weight_new[i] - req_datas[req.sender][i]) / (float)(req_metas.size() - 1);
            }
          }
          server->Response(req);
        }

        req_metas.clear();

        std::cout << "current weight: " << std::endl;
        for (size_t i = 0; i < d_; ++i)
        {
          weight_old[i] = weight_new[i];
          weight_new[i] = 0;
          std::cout << weight_old[i] << " ";
        }
        std::cout << std::endl;
        string str = "result/";
        str += string(ps::Environment::Get()->find("FEATURE_SIZE")) + "_";
        str += std::to_string(round_) + ".model";
        rr::SaveModel(str, weight_old, d_);

        round_++;
        std::cout << std::endl;
        std::cout << "start round " << round_ << std::endl;
      }
    }
    else
    {
      KVPairs<Val> res_data;

      res_data.keys = req_data.keys;
      res_data.vals.resize(d_, 0);

      if (weight_r.find(req_meta.sender) != weight_r.end())
      {
        for (size_t i = 0; i < d_; ++i)
        {
          res_data.vals[i] = weight_r[req_meta.sender][i];
        }
      }
      server->Response(req_meta, res_data);
    }
  }

  float zeta_;
  size_t d_;
  KVServer<float> *ps_server_;
  size_t round_;

  std::vector<KVMeta> req_metas;
  std::unordered_map<size_t, std::vector<float>> req_datas;

  float *weight_new;
  float *weight_old;
  std::unordered_map<size_t, std::vector<float>> weight_r;
};

void StartServer()
{
  if (!IsServer())
  {
    return;
  }
  auto server = new KVDCServer<float>(rr::ToFloat(ps::Environment::Get()->find("ZETA")),
                                      rr::ToInt(ps::Environment::Get()->find("FEATURE_SIZE")));
  RegisterExitCallback([server]() { delete server; });
}

void RunWorker(int argc, char *argv[])
{
  if (!IsWorker() || argc < 2)
    return;
  KVWorker<float> kv(0, 0);

  // 加载数据
  int n = rr::ToInt(ps::Environment::Get()->find("SAMPLE_SIZE"));
  int d = rr::ToInt(ps::Environment::Get()->find("FEATURE_SIZE"));
  float lambda = rr::ToFloat(ps::Environment::Get()->find("LAMBDA"));
  float gamma = rr::ToFloat(ps::Environment::Get()->find("GAMMA"));
  std::cout << "sample size: " << n << std::endl;

  rr::Dataset dataset_(n, d);
  rr::LoadData(argv[1], dataset_);
  std::vector<Key> keys(d);
  std::vector<float> vals(d);

  // 计算inv(A),b,c0
  rr::RidgeRegression rr(dataset_, lambda, gamma);

  // push
  for (size_t j = 0; j < rr::ToInt(ps::Environment::Get()->find("MAX_ITERATION")) && !vals.empty(); ++j)
  {
    float *current_w = rr.Getw();
    for (size_t i = 0; i < d; ++i)
    {
      keys[i] = i;
      vals[i] = current_w[i];
    }

    bool finished = true;
    // Actual update
    kv.Wait(kv.Push(keys, vals));
    kv.Wait(kv.Pull(keys, &vals));

    std::cout << "Responsed wR : " << std::endl;
    for (size_t i = 0; i < d; ++i)
      std::cout << vals[i] << " ";
    std::cout << std::endl;

    for (size_t i = 0; i < d; ++i)
    {
      if (vals[i] != 0)
        finished = false;
    }

    if (finished)
      break;

    rr.SetwR_(vals);
  }
}

int main(int argc, char *argv[])
{
  // start system
  Start(0);
  // setup server nodes
  StartServer();
  // run worker nodes
  RunWorker(argc, argv);
  // stop system
  Finalize(0, true);

  return 0;
}
