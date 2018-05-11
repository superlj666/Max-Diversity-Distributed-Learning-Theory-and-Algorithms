ifdef config
include $(config)
endif

ifndef CXX
CXX = g++
endif

ifndef DEPS_PATH
DEPS_PATH = $(shell pwd)/ps-lite/deps
endif

# INCPATH = -I./src -I./include -I./ps-lite/include -I$(CPLUS_INCLUDE_PATH)
# CFLAGS = -std=c++11 -msse2 -fPIC -O3 -ggdb -Wall -finline-functions -lmkl_core -lmkl_intel_lp64 -lmkl_intel_thread -liomp5 -lpthread -lm $(INCPATH) $(ADD_CFLAGS)
# LD_FLAGS = -Wl,-rpath,$DEPS_PATH/lib -L$DEPS_PATH/lib -lprotobuf-lite -lzmq -pthread -L/opt/intel/mkl/lib/intel64 -L/opt/intel/lib/intel64


# $(CXX) $(INCPATH) -std=c++0x -MM -MT build/ridge_regression.o src/ridge_regression.cc >build/ridge_regression.d
# $(CXX) $(CFLAGS) $(INCPATH) -c src/ridge_regression.cc -o build/ridge_regression.o
# ar crv build/librr.a build/ridge_regression.o
# $(CXX) -std=c++0x $(CFLAGS) $(INCPATH) -MM -MT tests/test_ridge_regression tests/test_ridge_regression.cc >tests/test_ridge_regression.d
# $(CXX) -std=c++0x $(CFLAGS) $(INCPATH) -o tests/test_ridge_regression tests/test_ridge_regression.cc build/librr.a ./ps-lite/build/libps.a $(LDFLAGS)


## simplization version
	g++ -I./src -I./include -I./ps-lite/include -I$CPLUS_INCLUDE_PATH -std=c++0x -MM -MT build/ridge_regression.o src/ridge_regression.cc >build/ridge_regression.d
	g++ -std=c++11 -lmkl_core -lmkl_intel_lp64 -lmkl_intel_thread -liomp5 -lpthread -lm -I./src -I./include -I./ps-lite/include -I$CPLUS_INCLUDE_PATH  -c src/ridge_regression.cc -o build/ridge_regression.o
	ar crv build/librr.a build/ridge_regression.o
	g++ -std=c++0x -std=c++11 -lmkl_core -lmkl_intel_lp64 -lmkl_intel_thread -liomp5 -lpthread -lm -I./src -I./include -I./ps-lite/include -I$CPLUS_INCLUDE_PATH -MM -MT tests/test_ridge_regression tests/test_ridge_regression.cc >tests/test_ridge_regression.d
	g++ -std=c++0x -std=c++11 -lmkl_core -lmkl_intel_lp64 -lmkl_intel_thread -liomp5 -lpthread -lm -I./src -I./include -I./ps-lite/include -I$CPLUS_INCLUDE_PATH -o tests/test_ridge_regression tests/test_ridge_regression.cc build/librr.a ./ps-lite/build/libps.a -Wl,-rpath,./ps-lite/deps/lib  -L./ps-lite/deps/lib  -lprotobuf-lite -lzmq -pthread -L/opt/intel/mkl/lib/intel64 -L/opt/intel/lib/intel64