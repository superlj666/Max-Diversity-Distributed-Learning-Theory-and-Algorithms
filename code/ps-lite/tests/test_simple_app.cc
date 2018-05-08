#include "ps/ps.h"
#include <iostream>
#include <unistd.h>
using namespace ps;

int num = 0;

void ReqHandle(const SimpleData& req, SimpleApp* app) {
  CHECK_EQ(req.head, 1);
  CHECK_EQ(req.body, "test");
  app->Response(req);
  ++ num;
  std::cout << "cur num: " << num << std::endl;
}

int main(int argc, char *argv[]) {
  int n = 100;
  Start(0);
  SimpleApp app(0, 0);
  app.set_request_handle(ReqHandle);

  std::cout << "s1: " << std::endl;
  if (IsScheduler()) {
    std::cout << "s2: " << std::endl;
    std::vector<int> ts;
    for (int i = 0; i < n; ++i) {
      int recver = kScheduler + kServerGroup + kWorkerGroup;
      ts.push_back(app.Request(1, "test", 4));
    }
    
    for (int t : ts) {
      app.Wait(t);
    }
  }

  std::cout << "final result of " << (IsScheduler()?"scheduler_":(IsWorker()?"worker_":"server_")) << MyRank() << " is " << num << std::endl;
  Finalize(0, true);

  if(IsWorker()){
    CHECK_EQ(num, n);
  }
  return 0;
}
