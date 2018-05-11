#include <mkl.h>
#include <math.h>
#include "ps/ps.h"
#include "kernel_ridge_regression.h"
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

    test_file_path_ = argv[1];
    test_file_path_ += "test";
    train_file_path_ = argv[1];
    train_file_path_ += "train_0";
    test_sample_size_ = rr::ToInt(ps::Environment::Get()->find("TEST_SAMPLE_SIZE"));
    train_sample_size_ = rr::ToInt(ps::Environment::Get()->find("TRAIN_SAMPLE_SIZE"));
    d_ = rr::ToInt(ps::Environment::Get()->find("FEATURE_SIZE"));

    weight_new = new float[train_sample_size_]();
    weight_old = new float[train_sample_size_]();
    round_ = 0;

    cout << "start round " << round_ << endl;

    using namespace placeholders;
    ps_server_->set_request_handle(
        bind(&KVDCServer::ReqHandle, this, _1, _2, _3));
  }
  float getMSE()
  {
    cout << "train_file_path_: " <<  train_file_path_  << endl;
    cout << "test_file_path_: " <<  test_file_path_  << endl;

    rr::Dataset test_data(test_sample_size_, d_);
    rr::LoadData(test_file_path_, test_data);
    rr::Dataset train_data(train_sample_size_, d_);
    rr::LoadData(train_file_path_, train_data);

    cout << "n: " <<  test_data.n  << endl;
    float *predict = new float[test_data.n]();

    cout << "Predict and MSE:" << endl;
    float workers = rr::ToFloat(ps::Environment::Get()->find("DMLC_NUM_WORKER"));
    for (int i = 0; i < train_sample_size_; ++i)
    {
      weight_old[i] = weight_old[i] / workers;
    }
    rr::PrintMatrix(train_sample_size_, 1, weight_old);
    rr::KRRPredict(train_data, test_data, weight_old, predict, rr::ToInt(ps::Environment::Get()->find("SIGMA")));
    float mse = rr::MSE(test_data, predict);
    cout << "MSE :" << mse << endl;

    delete predict;
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
      CHECK_EQ(train_sample_size_, req_data.vals.size());

      // Update
      req_metas.push_back(req_meta);
      if (req_datas.find(req_meta.sender) == req_datas.end())
      {
        req_datas[req_meta.sender] = vector<float>(train_sample_size_);
        weight_r[req_meta.sender] = vector<float>(train_sample_size_);
      }
      cout << "push data from worker " << req_meta.sender << endl;
      for (int i = 0; i < train_sample_size_; ++i)
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
        for (int i = 0; i < train_sample_size_; ++i)
        {
          var += pow(weight_new[i] - weight_old[i], 2);
        }
        var = sqrt(var);
        cout << "var: " << var << endl;

        for (auto &req : req_metas)
        {
          if (var <= zeta_ / train_sample_size_)
          {
            weight_r.clear();
          }
          else
          {
            for (int i = 0; i < train_sample_size_; ++i)
            {
              weight_r[req.sender][i] = (weight_new[i] - req_datas[req.sender][i]) / (float)(req_metas.size() - 1);
            }
          }
          server->Response(req);
        }

        req_metas.clear();

        cout << "current weight: " << endl;
        for (int i = 0; i < train_sample_size_; ++i)
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
      res_data.vals.resize(train_sample_size_, 0);

      if (weight_r.find(req_meta.sender) != weight_r.end())
      {
        for (int i = 0; i < train_sample_size_; ++i)
        {
          res_data.vals[i] = weight_r[req_meta.sender][i];
        }
      }
      server->Response(req_meta, res_data);
    }
  }

  string test_file_path_;
  string train_file_path_;
  int test_sample_size_;
  int train_sample_size_;
  int d_;
  float zeta_;
  KVServer<float> *ps_server_;
  int round_;

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
  int n = rr::ToInt(argv[2]);
  int d = rr::ToInt(argv[3]);
  float lambda = rr::ToFloat(ps::Environment::Get()->find("LAMBDA"));
  float gamma = rr::ToFloat(ps::Environment::Get()->find("GAMMA"));

  rr::Dataset dataset_(n, d);
  rr::LoadData(argv[1], dataset_);
  vector<Key> keys(n);
  vector<float> vals(n);

  // 计算inv(A),b,c0
  rr::KernelRidgeRegression krr(dataset_, lambda, gamma);

  // push
  for (int j = 0; j < rr::ToInt(ps::Environment::Get()->find("MAX_ITERATION")) && !vals.empty(); ++j)
  {
    float *current_w = krr.Getw();
    for (int i = 0; i < n; ++i)
    {
      keys[i] = i;
      vals[i] = current_w[i];
    }

    bool finished = true;
    // Actual update
    kv.Wait(kv.Push(keys, vals));
    kv.Wait(kv.Pull(keys, &vals));

    cout << "Responsed wR : " << endl;
    for (int i = 0; i < n; ++i)
      cout << vals[i] << " ";
    cout << endl;

    for (int i = 0; i < n; ++i)
    {
      if (vals[i] != 0)
        finished = false;
    }

    if (finished)
      break;

    krr.SetwR_(vals);
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
