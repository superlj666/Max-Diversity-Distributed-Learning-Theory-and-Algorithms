TEST_SRC = $(wildcard tests/test_*.cc)
TEST = $(patsubst tests/test_%.cc, tests/test_%, $(TEST_SRC))

LDFLAGS = -Wl,-rpath,$(DEPS_PATH)/lib $(PS_LDFLAGS_SO) -pthread -L/opt/intel/mkl/lib/intel64 -L/opt/intel/lib/intel64
tests/% : tests/%.cc build/libps.a
	$(CXX) -std=c++0x $(CFLAGS) -MM -MT tests/$* $< >tests/$*.d
	$(CXX) -std=c++0x $(CFLAGS) -o $@ $(filter %.cc %.a, $^) $(LDFLAGS)

-include tests/*.d

g++ -std=c++0x -std=c++11 -msse2 -fPIC -O3 -ggdb -Wall -finline-functions -I./src -I./include -I/home/bd-dev/lijian/NIPS_2018/ps-lite/deps/include -MM -MT tests/ps tests/ps.cc >tests/ps.d
g++ -std=c++0x -std=c++11 -msse2 -fPIC -O3 -ggdb -Wall -finline-functions -I./src -I./include -I/home/bd-dev/lijian/NIPS_2018/ps-lite/deps/include -o tests/ps tests/ps.cc build/libps.a -Wl,-rpath,/home/bd-dev/lijian/NIPS_2018/ps-lite/deps/lib -L/home/bd-dev/lijian/NIPS_2018/ps-lite/deps/lib -lprotobuf-lite -lzmq -pthread