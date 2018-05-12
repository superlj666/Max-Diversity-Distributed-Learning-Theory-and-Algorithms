#include <mkl.h>
#include <math.h>
#include "ps/ps.h"
#include "dist_krr.h"
#include "time.h"
#define FILE_ID_REQ 111
#define STOP_TRAIN 222
#define PREDICT_REQ 333
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
    train_size_ = rr::ToInt(ps::Environment::Get()->find("TRAIN_SAMPLE_SIZE"));
    workers_ = rr::ToFloat(ps::Environment::Get()->find("DMLC_NUM_WORKER"));
    zeta_ = rr::ToFloat(ps::Environment::Get()->find("ZETA"));
    round_ = 0;
    var_ = 0;
    finished_ = false;

    mean_predict = new float[rr::ToInt(ps::Environment::Get()->find("TEST_SAMPLE_SIZE"))]();

    all_weight_.resize(workers_ * train_size_, 0);
    cout << "start round " << round_ << endl;

    using namespace placeholders;
    ps_server_->set_request_handle(
        bind(&KVDCServer::ReqHandle, this, _1, _2, _3));
  }

  ~KVDCServer()
  {
    delete mean_predict;
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
    cout << "request size: " << req_data.vals.size() << endl;
    cout << "cmd: " << req_meta.cmd << endl;

    req_metas_.push_back(req_meta);
    if (req_meta.push)
    {
      if (req_meta.cmd == FILE_ID_REQ)
      {
        cout << "Sender ID: " << req_meta.sender << endl;
        cout << "Sender file rank: " << (int)req_data.vals[0] << endl;
        id_file_rank_[req_meta.sender] = (int)req_data.vals[0];
        cout << req_metas_.size() << " " << NumWorkers() << endl;

        if (req_metas_.size() == (int)NumWorkers())
        {
          for (auto &req : req_metas_)
          {
            cout << "response id mapping..." << endl;
            server->Response(req);
          }
          req_metas_.clear();
        }
      }
      else if (req_meta.cmd == PREDICT_REQ)
      {
        cout << "Predict size: " << req_data.vals.size() << endl;
        for (int i = 0; i < req_data.vals.size(); ++i)
        {
          mean_predict[i] = req_data.vals[i];
        }

        string predict_save = "result/";
        predict_save += clock();
        rr::SaveModel(predict_save.c_str(), mean_predict, req_data.vals.size(), 1);

        if (req_metas_.size() == (int)NumWorkers())
        {
          for (auto &req : req_metas_)
          {
            cout << "response id mapping..." << endl;
            server->Response(req);
          }
          req_metas_.clear();
        }
      }
      else
      {
        // Update
        cout << "all weight size: " << all_weight_.size() << endl;
        for (int i = 0; i < req_data.vals.size(); ++i)
        {
          int index = id_file_rank_[req_meta.sender] * train_size_ + i;
          var_ += pow(all_weight_[index] - req_data.vals[i], 2);
          all_weight_[i] = req_data.vals[i];
        }

        cout << req_metas_.size() << " " << NumWorkers() << endl;
        // Wait all push to Update wr and response
        if (req_metas_.size() == (int)NumWorkers())
        {
          // Compute variation
          cout << "var_: " << var_ << endl;
          var_ = sqrt(var_ / (float)all_weight_.size());
          if (var_ <= zeta_)
          {
            finished_ = true;
          }

          cout << endl;
          round_++;
          cout << "start round " << round_ << endl;

          for (auto &req : req_metas_)
          {
            cout << "response..." << endl;
            server->Response(req);
          }
          req_metas_.clear();
          var_ = 0;

          cout << endl;
        }
      }
    }
    else
    {
      cout << "pull request " << endl;
      KVPairs<Val> res_data;

      if (!finished_)
      {
        res_data.keys = req_data.keys;
        res_data.vals = SArray<float>(all_weight_);
        cout << res_data.vals.size() << "---" << endl;
      }

      res_data.keys = req_data.keys;
      res_data.vals.resize(workers_ * train_size_, 0);
      server->Response(req_meta, res_data);

      cout << "response pull...." << endl;
    }
    cout << endl;
  }

  KVServer<float> *ps_server_;
  int train_size_;
  int workers_;
  float zeta_;
  float var_;
  int round_;
  bool finished_;

  vector<KVMeta> req_metas_;
  unordered_map<int, int> id_file_rank_;
  vector<float> all_weight_;
  float *mean_predict;
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
  cout << "start worker..." << endl;

  int all_size = rr::ToInt(ps::Environment::Get()->find("TRAIN_SAMPLE_SIZE")) *
                 rr::ToFloat(ps::Environment::Get()->find("DMLC_NUM_WORKER"));
  vector<Key> keys(1);
  vector<float> vals(1);

  // 0. Tell server file rank and its id
  keys[0] = 1;
  vals[0] = rr::ToInt(argv[1]);
  kv.Wait(kv.Push(keys, vals, {}, FILE_ID_REQ, nullptr));
  cout << "finish 0..." << endl;

  // 1. Loading Training Data (mainly for labels)
  int n = rr::ToInt(ps::Environment::Get()->find("TRAIN_SAMPLE_SIZE"));
  int d = rr::ToInt(ps::Environment::Get()->find("FEATURE_SIZE"));
  float lambda = rr::ToFloat(ps::Environment::Get()->find("LAMBDA"));
  float gamma = rr::ToFloat(ps::Environment::Get()->find("GAMMA"));
  string file_path(ps::Environment::Get()->find("DATA_PATH"));
  file_path += "train_";
  if (rr::ToInt(argv[1]) > 9)
  {
    file_path += argv[1];
  }
  else
  {
    file_path += "0";
    file_path += argv[1];
  }
  cout << "file_path: " << file_path << endl;
  rr::Dataset train_data(n, d);
  rr::LoadData(file_path, train_data);

  // 2. Loading Testing Data (mainly for labels)
  int n_test = rr::ToInt(ps::Environment::Get()->find("TEST_SAMPLE_SIZE"));
  string test_file_path(ps::Environment::Get()->find("DATA_PATH"));
  test_file_path += "test_00";
  cout << "test_file_path: " << test_file_path << endl;
  rr::Dataset test_data(n_test, d);
  rr::LoadData(test_file_path, test_data);

  // 3.1 Loading Self Train Kernel
  string self_kernel_name = "train_";
  if (rr::ToInt(argv[1]) > 9)
  {
    self_kernel_name += argv[1];
  }
  else
  {
    self_kernel_name += "0";
    self_kernel_name += argv[1];
  }
  self_kernel_name = self_kernel_name + "-" + self_kernel_name;
  string self_kernel_path = ps::Environment::Get()->find("KERNEL_PATH") + self_kernel_name;
  cout << self_kernel_path << endl;
  rr::KernelData selfKernel(self_kernel_path, n, n);

  // 3.2 Loading Self Test Kernel
  string self_test_kernel_name = "train_";
  if (rr::ToInt(argv[1]) > 9)
  {
    self_test_kernel_name += argv[1];
  }
  else
  {
    self_test_kernel_name += "0";
    self_test_kernel_name += argv[1];
  }
  self_test_kernel_name = self_test_kernel_name + "-test_00";
  string self_test_kernel_path = ps::Environment::Get()->find("KERNEL_PATH") + self_test_kernel_name;
  cout << self_test_kernel_path << endl;
  rr::KernelData selfTestKernel(self_test_kernel_path, n, n);

  // 4. Computing w0 and push
  rr::DistKRR distKRR(selfKernel, train_data.label, lambda, gamma);

  keys.resize(n, 0);
  vals.resize(n, 0);
  for (int j = 0; j < rr::ToInt(ps::Environment::Get()->find("MAX_ITERATION")) && !vals.empty(); ++j)
  {
    cout << "start worker...3" << endl;
    float *current_w = distKRR.Getw();

    for (int i = 0; i < n; ++i)
    {
      keys[i] = i;
      vals[i] = current_w[i];
    }

    // 5. Push weight to server
    bool finished = true;
    kv.Wait(kv.Push(keys, vals));

    // 6. Receiving all weight
    vals.resize(all_size, 0);
    cout << keys.size() << " " << vals.size() << endl;
    kv.Wait(kv.Pull(keys, &vals));

    cout << "Responsed wR : " << endl;
    for (int i = 0; i < all_size; ++i)
      cout << vals[i] << " ";
    cout << endl;

    for (int i = 0; i < n; ++i)
    {
      if (vals[i] != 0)
        finished = false;
    }

    //  7. Local weight predict
    if (finished)
    {
      float *predict = test_data.label;
      KernelPredict(selfTestKernel, current_w, predict);
      keys.resize(n_test, 0);
      vals.resize(n_test);
      for (int i = 0; i < n_test; ++i)
      {
        vals[i] = predict[i];
        cout << vals[i] << " ";
      }
      kv.Push(keys, vals, {}, PREDICT_REQ, nullptr);
      return;
    }

    // 8. Update w
    distKRR.SetwR_(vals);
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