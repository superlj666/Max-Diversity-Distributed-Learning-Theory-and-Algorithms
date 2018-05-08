#include <mkl.h>
#include <math.h>
#include "ps/ps.h"
using namespace ps;

/**
 * \brief a req_handle to process request in dc method
 */
template <typename Val>
struct KVServerDCHandle
{
public:
  void operator()(
      const KVMeta &req_meta, const KVPairs<Val> &req_data, KVServer<Val> *server)
  {
    size_t d = req_data.keys.size();
    KVPairs<Val> res;
    if (req_meta.push)
    {
      CHECK_EQ(d, req_data.vals.size());

      // Record
      reqs.push_back(make_pair(req_meta, req_data));
      float var = 0;
      for (size_t i = 0; i < d; ++i)
      {
        newer_w.push_back(req_data.vals[i]);
        if (!older_w.empty())
        {
          var += pow(newer_w[i] - older_w[i], 2);
        }
        var = sqrt(var);
      }

      // Response
      if (reqs.size() == (size_t)ps::NumWorkers())
      {
        for (auto &req : reqs)
        {
          if (var <= zeta)
          {
            req.keys.clear();
            req.vals.clear();
          }
          else
          {
            for (size_t i = 0; i < d; ++i)
            {
              req.second.vals[i] = (newer_w[i] - req.second.vals[i]) / (float)(reqs.size() - 1);
            }
            older_w=newer_w;
            newer_w.clear();
          }
          server->Response(req.first, req.second);
        }
        reqs.clear();
        w.clear();
      }
    }
    else
    {
      // TODO
    }
  }
  static float zeta=1e-5;

  std::vector<Key, Val> older_w;
  std::vector<Key, Val> newer_w;
  std::vector<pair<KVMeta &, KVPairs<Val> &>> reqs;
};

template <typename Val>
struct KVWorkerDCHandle
{
public:
  void operator()(
      const KVMeta &req_meta, const KVPairs<Val> &req_data, KVServer<Val> *server)
  {
    wr = req_data.vals;
  }

  static std::vector<float> &wr;
};

void StartServer()
{
  if (!IsServer())
  {
    return;
  }
  auto server = new KVServer<float>(0);
  // KVServerDCHandle<float>::zeta = 1e-5;

  server->set_request_handle(KVServerDCHandle<float>());
  RegisterExitCallback([server]() { delete server; });
}

void RunWorker()
{
  if (!IsWorker())
    return;
  KVWorker<float> kv(0, 0);
  int max_iter = 10;

  // init
  int num = 10;
  std::vector<Key> keys(num);
  std::vector<float> vals(num);
  // kv.set_response_handle(KVWorkerDCHandle<float>());

  for (int i = 0; i < num; ++i)
  {
    keys[i] = i;
    vals[i] = i;
  }

  // push
  for (int i = 0; i < max_iter && !vals.empty(); ++i)
  {
    // TODO. Actual update
    kv.Wait(kv.Push(keys, vals));
  }
}

int main(int argc, char *argv[])
{
  // start system
  Start(0);
  // setup server nodes
  StartServer();
  // run worker nodes
  RunWorker();
  // stop system
  Finalize(0, true);
  return 0;
}
