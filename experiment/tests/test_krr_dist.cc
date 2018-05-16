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
  KVDCServer(float *all_weight, float *mean_predict)
  {
    cout << "start server..." << endl;

    ps_server_ = new KVServer<float>(0);
    train_size_ = rr::ToInt(ps::Environment::Get()->find("TRAIN_SAMPLE_SIZE"));
    workers_ = rr::ToFloat(ps::Environment::Get()->find("DMLC_NUM_WORKER"));
    zeta_ = rr::ToFloat(ps::Environment::Get()->find("ZETA"));
    round_ = 0;
    var_ = 0;
    finished_ = false;
    total_size = workers_ * train_size_;

    mean_predict_ = mean_predict;
    all_weight_ = all_weight;

    using namespace placeholders;
    ps_server_->set_request_handle(
        bind(&KVDCServer::ReqHandle, this, _1, _2, _3));

    cout << "start round " << round_ << endl;
  }

  ~KVDCServer()
  {
    delete all_weight_;
    delete mean_predict_;
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

    if (req_meta.push)
    {
      req_metas_.push_back(req_meta);
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
          mean_predict_[i] += req_data.vals[i] / workers_;
        }

        if (req_metas_.size() == (int)NumWorkers())
        {

          string predict_save = "result/";
          predict_save += to_string(clock());
          cout << predict_save << endl;
          if (rr::SaveModel(predict_save.c_str(), mean_predict_, req_data.vals.size(), 1))
          {
            cout << "Save predict successly in " << predict_save << endl;
          }
          else
          {
            cout << "Save predict failed." << endl;
          }
          int n_test = rr::ToInt(ps::Environment::Get()->find("TEST_SAMPLE_SIZE"));
          int d = rr::ToInt(ps::Environment::Get()->find("FEATURE_SIZE"));
          string test_file_path(ps::Environment::Get()->find("DATA_PATH"));
          test_file_path += "test_00";
          cout << "test_file_path: " << test_file_path << endl;
          rr::Dataset test_data(n_test, d);
          rr::LoadData(test_file_path, test_data);

          float mse = rr::MSE(test_data, mean_predict_);
          cout << "MSE :" << mse << endl;
          for (auto &req : req_metas_)
          {
            server->Response(req);
          }
          req_metas_.clear();

          return;
        }
      }
      else
      {
        // Update
        cout << "all weight size: " << total_size << endl;
        for (int i = 0; i < req_data.vals.size(); ++i)
        {
          int index = id_file_rank_[req_meta.sender] * train_size_ + i;
          var_ += pow(all_weight_[index] - req_data.vals[i], 2);
          all_weight_[index] = req_data.vals[i];
        }

        cout << req_metas_.size() << " " << NumWorkers() << endl;
        // Wait all push to Update wr and response
        if (req_metas_.size() == (int)NumWorkers())
        {
          // Compute variation
          var_ = sqrt(var_ / (float)total_size);
          cout << "var_: " << var_ << endl;
          if (var_ <= zeta_ || round_ == rr::ToInt(ps::Environment::Get()->find("MAX_ITERATION")) - 1)
          {
            finished_ = true;
          }

          round_++;
          cout << "start round " << round_ << endl;

          for (auto &req : req_metas_)
          {
            cout << "response..." << endl;
            server->Response(req);
          }
          var_ = 0;
          req_metas_.clear();
        }
      }
    }
    else
    {
      cout << "pull request " << endl;
      KVPairs<Val> res_data;
      res_data.keys = req_data.keys;
      res_data.vals.resize(total_size, 0);

      if (!finished_)
      {
        for (int i = 0; i < total_size; ++i)
        {
          res_data.vals[i] = all_weight_[i];
        }
      }

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
  float *all_weight_;
  float *mean_predict_;
  int total_size;
};

void StartServer(int argc, char *argv[])
{
  if (!IsServer())
    return;

  int train_size_ = rr::ToInt(ps::Environment::Get()->find("TRAIN_SAMPLE_SIZE"));
  int workers_ = rr::ToInt(ps::Environment::Get()->find("DMLC_NUM_WORKER"));
  int total_size = workers_ * train_size_;

  float *mean_predict = new float[rr::ToInt(ps::Environment::Get()->find("TEST_SAMPLE_SIZE"))]();
  float *all_weight = new float[total_size]();

  auto server = new KVDCServer<float>(all_weight, mean_predict);
  RegisterExitCallback([server]() { delete server; });
}

void RunWorker(int argc, char *argv[])
{
  if (!IsWorker())
    return;
  KVWorker<float> kv(0, 0);
  cout << "start worker..." << endl;

  int n = rr::ToInt(ps::Environment::Get()->find("TRAIN_SAMPLE_SIZE"));
  int d = rr::ToInt(ps::Environment::Get()->find("FEATURE_SIZE"));
  int workers = rr::ToFloat(ps::Environment::Get()->find("DMLC_NUM_WORKER"));
  float lambda = rr::ToFloat(ps::Environment::Get()->find("LAMBDA"));
  float gamma = rr::ToFloat(ps::Environment::Get()->find("GAMMA"));
  string file_path(ps::Environment::Get()->find("DATA_PATH"));
  int file_id = rr::ToInt(argv[1]);
  int all_size = n * workers;
  vector<Key> keys(1);
  vector<float> vals(1);

  // 0. Tell server file rank and its id
  keys[0] = 1;
  vals[0] = file_id;
  kv.Wait(kv.Push(keys, vals, {}, FILE_ID_REQ, nullptr));
  cout << "finish 0..." << endl;

  // 1. Loading Training Data (mainly for labels)
  file_path += "train_";
  if (file_id > 9)
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

  // 2. Loading Testing Data (mainly for size)
  int n_test = rr::ToInt(ps::Environment::Get()->find("TEST_SAMPLE_SIZE"));
  string test_file_path(ps::Environment::Get()->find("DATA_PATH"));
  test_file_path += "test_00";
  cout << "test_file_path: " << test_file_path << endl;
  rr::Dataset test_data(n_test, d);
  rr::LoadData(test_file_path, test_data);

  // 3 Loading Self Train Kernel
  string self_kernel_name = "train_";
  if (file_id > 9)
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

  // 4. Computing w0 and push
  rr::DistKRR distKRR(selfKernel, train_data.label, lambda, gamma);

  while (true)
  {
    float *current_w = distKRR.Getw();

    keys.resize(n, 0);
    vals.resize(n, 0);
    for (int i = 0; i < n; ++i)
    {
      keys[i] = i;
      vals[i] = current_w[i];
    }

    // 5. Push weight to server
    bool finished = true;
    kv.Wait(kv.Push(keys, vals));

    // 6. Pull and Receiving all weight
    keys.resize(all_size, 0);
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
      cout << "Local weight predict------>" << endl;
      float *predict = new float[test_data.n]();
      string self_test_kernel_path = ps::Environment::Get()->find("KERNEL_PATH");
      self_test_kernel_path += "train_";
      self_test_kernel_path += rr::ID2string(file_id);
      self_test_kernel_path += "-test_00";

      cout << self_test_kernel_path << endl;
      rr::KernelData selfTestKernel(self_test_kernel_path, n, n_test);
      // rr::PrintMatrix(n, n_test, selfTestKernel.kernel);

      rr::KernelPredict(selfTestKernel, current_w, predict);
      cout << test_data.n << endl;
      string save_path = "result/test_tmp_";
      save_path += argv[1];
      rr::SaveModel(save_path.c_str(), predict, test_data.n, 1);

      keys.resize(n_test, 0);
      vals.resize(n_test, 0);
      for (int i = 0; i < n_test; ++i)
      {
        vals[i] = predict[i];
        cout << vals[i] << " ";
      }
      kv.Wait(kv.Push(keys, vals, {}, PREDICT_REQ, nullptr));

      delete predict;
      return;
    }

    // 8. Update w
    cout << "Update w------>" << endl;
    float *wR_ = new float[n]();
    float *other_w = new float[n]();
    float *predict = new float[test_data.n]();

    cout << "workers" << workers -1 << endl;

    for (int i = 0; i < workers; ++i)
    {
      if (i != file_id)
      {
        for (int j = 0; j < n; ++j)
        {
          other_w[j] = vals[i * n + j];
        }

        string cross_kernel_path = ps::Environment::Get()->find("KERNEL_PATH");
        cross_kernel_path += "train_";
        cross_kernel_path += rr::ID2string(i);
        cross_kernel_path += "-train_";
        cross_kernel_path += rr::ID2string(file_id);

        cout << cross_kernel_path << endl;
        rr::KernelData selfTestKernel(cross_kernel_path, n, n);
        rr::KernelPredict(selfTestKernel, other_w, predict);
        
        cout << "weight from worker " << i << endl;
        rr::PrintMatrix(1, n, other_w);
        for (int j = 0; j < n; ++j)
        {
          wR_[j] += predict[j] / (float)(workers - 1);
        }
      }
    }

    cout << "mean predict:" << endl;
    rr::PrintMatrix(1, n, wR_);
    distKRR.SetwR_(wR_);
    delete wR_;
    delete other_w;
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
