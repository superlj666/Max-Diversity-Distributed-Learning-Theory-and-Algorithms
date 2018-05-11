# ridge_regression
g++ -I./src -I./include -I./ps-lite/include -I$CPLUS_INCLUDE_PATH -std=c++0x -MM -MT build/ridge_regression.o src/ridge_regression.cc >build/ridge_regression.d
g++ -std=c++11 -lmkl_core -lmkl_intel_lp64 -lmkl_intel_thread -liomp5 -lpthread -lm -I./src -I./include -I./ps-lite/include -I$CPLUS_INCLUDE_PATH  -c src/ridge_regression.cc -o build/ridge_regression.o
ar crv build/librr.a build/ridge_regression.o

# kernel_ridge_regression
g++ -I./src -I./include -I./ps-lite/include -I$CPLUS_INCLUDE_PATH -std=c++0x -MM -MT build/kernel_ridge_regression.o src/kernel_ridge_regression.cc >build/kernel_ridge_regression.d
g++ -std=c++11 -lmkl_core -lmkl_intel_lp64 -lmkl_intel_thread -liomp5 -lpthread -lm -I./src -I./include -I./ps-lite/include -I$CPLUS_INCLUDE_PATH  -c src/kernel_ridge_regression.cc -o build/kernel_ridge_regression.o
ar crv build/librr.a build/kernel_ridge_regression.o

# tmp
g++ -std=c++0x -std=c++11 -lmkl_core -lmkl_intel_lp64 -lmkl_intel_thread -liomp5 -lpthread -lm -I./src -I./include -I./ps-lite/include -I$CPLUS_INCLUDE_PATH -MM -MT tests/test_tmp tests/test_tmp.cc >tests/test_tmp.d
g++ -std=c++0x -std=c++11 -lmkl_core -lmkl_intel_lp64 -lmkl_intel_thread -liomp5 -lpthread -lm -I./src -I./include -I./ps-lite/include -I$CPLUS_INCLUDE_PATH -o tests/test_tmp tests/test_tmp.cc build/librr.a ./ps-lite/build/libps.a -Wl,-rpath,./ps-lite/deps/lib  -L./ps-lite/deps/lib  -lprotobuf-lite -lzmq -pthread -L/opt/intel/mkl/lib/intel64 -L/opt/intel/lib/intel64

# gaussian_kernel
g++ -std=c++0x -std=c++11 -lmkl_core -lmkl_intel_lp64 -lmkl_intel_thread -liomp5 -lpthread -lm -I./src -I./include -I./ps-lite/include -I$CPLUS_INCLUDE_PATH -MM -MT tests/test_gaussian_kernel tests/test_gaussian_kernel.cc >tests/test_gaussian_kernel.d
g++ -std=c++0x -std=c++11 -lmkl_core -lmkl_intel_lp64 -lmkl_intel_thread -liomp5 -lpthread -lm -I./src -I./include -I./ps-lite/include -I$CPLUS_INCLUDE_PATH -o tests/test_gaussian_kernel tests/test_gaussian_kernel.cc build/librr.a ./ps-lite/build/libps.a -Wl,-rpath,./ps-lite/deps/lib  -L./ps-lite/deps/lib  -lprotobuf-lite -lzmq -pthread -L/opt/intel/mkl/lib/intel64 -L/opt/intel/lib/intel64


# rr_central
g++ -std=c++0x -std=c++11 -lmkl_core -lmkl_intel_lp64 -lmkl_intel_thread -liomp5 -lpthread -lm -I./src -I./include -I./ps-lite/include -I$CPLUS_INCLUDE_PATH -MM -MT tests/test_rr_central tests/test_rr_central.cc >tests/test_rr_central.d
g++ -std=c++0x -std=c++11 -lmkl_core -lmkl_intel_lp64 -lmkl_intel_thread -liomp5 -lpthread -lm -I./src -I./include -I./ps-lite/include -I$CPLUS_INCLUDE_PATH -o tests/test_rr_central tests/test_rr_central.cc build/librr.a ./ps-lite/build/libps.a -Wl,-rpath,./ps-lite/deps/lib  -L./ps-lite/deps/lib  -lprotobuf-lite -lzmq -pthread -L/opt/intel/mkl/lib/intel64 -L/opt/intel/lib/intel64

# rr_dist
g++ -std=c++0x -std=c++11 -lmkl_core -lmkl_intel_lp64 -lmkl_intel_thread -liomp5 -lpthread -lm -I./src -I./include -I./ps-lite/include -I$CPLUS_INCLUDE_PATH -MM -MT tests/test_rr_dist tests/test_rr_dist.cc >tests/test_rr_dist.d
g++ -std=c++0x -std=c++11 -lmkl_core -lmkl_intel_lp64 -lmkl_intel_thread -liomp5 -lpthread -lm -I./src -I./include -I./ps-lite/include -I$CPLUS_INCLUDE_PATH -o tests/test_rr_dist tests/test_rr_dist.cc build/librr.a ./ps-lite/build/libps.a -Wl,-rpath,./ps-lite/deps/lib  -L./ps-lite/deps/lib  -lprotobuf-lite -lzmq -pthread -L/opt/intel/mkl/lib/intel64 -L/opt/intel/lib/intel64

# krr_central
g++ -std=c++0x -std=c++11 -lmkl_core -lmkl_intel_lp64 -lmkl_intel_thread -liomp5 -lpthread -lm -I./src -I./include -I./ps-lite/include -I$CPLUS_INCLUDE_PATH -MM -MT tests/test_krr_central tests/test_krr_central.cc >tests/test_krr_central.d
g++ -std=c++0x -std=c++11 -lmkl_core -lmkl_intel_lp64 -lmkl_intel_thread -liomp5 -lpthread -lm -I./src -I./include -I./ps-lite/include -I$CPLUS_INCLUDE_PATH -o tests/test_krr_central tests/test_krr_central.cc build/librr.a ./ps-lite/build/libps.a -Wl,-rpath,./ps-lite/deps/lib  -L./ps-lite/deps/lib  -lprotobuf-lite -lzmq -pthread -L/opt/intel/mkl/lib/intel64 -L/opt/intel/lib/intel64

# krr_dist
g++ -std=c++0x -std=c++11 -lmkl_core -lmkl_intel_lp64 -lmkl_intel_thread -liomp5 -lpthread -lm -I./src -I./include -I./ps-lite/include -I$CPLUS_INCLUDE_PATH -MM -MT tests/test_krr_dist tests/test_krr_dist.cc >tests/test_krr_dist.d
g++ -std=c++0x -std=c++11 -lmkl_core -lmkl_intel_lp64 -lmkl_intel_thread -liomp5 -lpthread -lm -I./src -I./include -I./ps-lite/include -I$CPLUS_INCLUDE_PATH -o tests/test_krr_dist tests/test_krr_dist.cc build/librr.a ./ps-lite/build/libps.a -Wl,-rpath,./ps-lite/deps/lib  -L./ps-lite/deps/lib  -lprotobuf-lite -lzmq -pthread -L/opt/intel/mkl/lib/intel64 -L/opt/intel/lib/intel64
