#include <mkl.h>
#include <math.h>
#include "ps/ps.h"
#include "ridge_regression.h"
using namespace ps;
using namespace std;

/**
 * \brief a req_handle to process request in dc method
 */
template <typename Val>
class KVDCServer
{
public:
  KVDCServer(int argc, char *argv[])
  {

    cout << "start server..." << endl;

    ps_server_ = new KVServer<float>(0);
    zeta_ = rr::ToFloat(ps::Environment::Get()->find("ZETA"));

    file_path_ += ps::Environment::Get()->find("DATA_PATH");
    file_path_ += "test_0000";
    cout << "test_file: " << file_path_ << endl;

    test_sample_size_ = rr::ToInt(ps::Environment::Get()->find("TEST_SAMPLE_SIZE"));
    d_ = rr::ToInt(ps::Environment::Get()->find("FEATURE_SIZE"));
    workers_ = rr::ToFloat(ps::Environment::Get()->find("DMLC_NUM_WORKER"));

    weight_new = new float[d_]();
    weight_old = new float[d_]();
    round_ = 0;

    cout << "start round " << round_ << endl;

    using namespace placeholders;
    ps_server_->set_request_handle(
        bind(&KVDCServer::ReqHandle, this, _1, _2, _3));
  }
  float getMSE()
  {
    rr::Dataset test_data(test_sample_size_, d_);
    rr::LoadData(file_path_, test_data);
    float *predict = new float[test_data.n]();

    cout << "Predict and MSE:" << endl;
    for (int i = 0; i < test_data.d; ++i)
    {
      weight_old[i] = weight_old[i] / workers_;
    }
    rr::Predict(test_data, weight_old, predict);
    rr::PrintMatrix(test_data.n, 1, predict);
    float mse = rr::MSE(test_data, predict);
    cout << mse << endl;

    return mse;
  }

  ~KVDCServer()
  {
    getMSE();
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
        req_datas[req_meta.sender] = vector<float>(d_);
        weight_r[req_meta.sender] = vector<float>(d_);
      }
      cout << "push data from worker " << req_meta.sender << endl;
      for (int i = 0; i < d_; ++i)
      {
        cout << req_data.vals[i] << " ";
        weight_new[i] += req_data.vals[i];
        req_datas[req_meta.sender][i] = req_data.vals[i];
      }
      cout << endl;

      // Wait all push to Update wr and response
      if (req_metas.size() == (int)NumWorkers())
      {
        float var = 0;
        for (int i = 0; i < d_; ++i)
        {
          var += pow(weight_new[i] / workers_ - weight_old[i] / workers_, 2);
        }
        var = sqrt(var / d_ / workers_);
        cout << "var: " << var << endl;

        for (auto &req : req_metas)
        {
          if (var <= zeta_)
          {
            weight_r.clear();
          }
          else
          {
            for (int i = 0; i < d_; ++i)
            {
              weight_r[req.sender][i] = (weight_new[i] - req_datas[req.sender][i]) / (float)(req_metas.size() - 1);
            }
          }
          server->Response(req);
        }

        req_metas.clear();

        cout << "current weight: " << endl;
        for (int i = 0; i < d_; ++i)
        {
          weight_old[i] = weight_new[i];
          weight_new[i] = 0;
          cout << weight_old[i] << " ";
        }
        cout << endl;
        round_++;

        cout << "start round " << round_ << endl;
      }
    }
    else
    {
      KVPairs<Val> res_data;

      res_data.keys = req_data.keys;
      res_data.vals.resize(d_, 0);

      if (weight_r.find(req_meta.sender) != weight_r.end())
      {
        for (int i = 0; i < d_; ++i)
        {
          res_data.vals[i] = weight_r[req_meta.sender][i];
        }
      }
      server->Response(req_meta, res_data);
    }
  }

  string file_path_;
  int test_sample_size_;
  int d_;
  float zeta_;
  KVServer<float> *ps_server_;
  int round_;
  float workers_;

  vector<KVMeta> req_metas;
  unordered_map<int, vector<float>> req_datas;

  float *weight_new;
  float *weight_old;
  unordered_map<int, vector<float>> weight_r;
};

void StartServer(int argc, char *argv[])
{
  if (!IsServer())
    return;
  auto server = new KVDCServer<float>(argc, argv);
  RegisterExitCallback([server]() { delete server; });
}

void RunWorker(int argc, char *argv[])
{
  if (!IsWorker())
    return;
  KVWorker<float> kv(0, 0);

  // 加载数据
  int n = rr::ToInt(ps::Environment::Get()->find("TRAIN_SAMPLE_SIZE"));
  int d = rr::ToInt(ps::Environment::Get()->find("FEATURE_SIZE"));
  float lambda = rr::ToFloat(ps::Environment::Get()->find("LAMBDA"));
  float gamma = rr::ToFloat(ps::Environment::Get()->find("GAMMA"));

  string file_path(ps::Environment::Get()->find("DATA_PATH"));
  file_path += "train_";
  file_path += rr::ID2string(rr::ToInt(argv[1]));
  cout << "file_path: " << file_path << endl;
  cout << "n: " << n << endl;
  cout << "d: " << d << endl;

  rr::Dataset dataset_(n, d);
  rr::LoadData(file_path, dataset_);
  vector<Key> keys(d);
  vector<float> vals(d);

  // 计算inv(A),b,c0
  rr::RidgeRegression rr(dataset_, lambda, gamma);

  cout << "finish loading" << endl;
  // push
  for (int j = 0; j < rr::ToInt(ps::Environment::Get()->find("MAX_ITERATION")) && !vals.empty(); ++j)
  {
    float *current_w = rr.Getw();
    for (int i = 0; i < d; ++i)
    {
      keys[i] = i;
      vals[i] = current_w[i];
    }

    bool finished = true;
    // Actual update
    kv.Wait(kv.Push(keys, vals));
    kv.Wait(kv.Pull(keys, &vals));

    cout << "Responsed wR : " << endl;
    for (int i = 0; i < d; ++i)
      cout << vals[i] << " ";
    cout << endl;

    for (int i = 0; i < d; ++i)
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
  StartServer(argc, argv);
  // run worker nodes
  RunWorker(argc, argv);
  // stop system
  Finalize(0, true);
  return 0;
}
