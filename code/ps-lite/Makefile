ifdef config
include $(config)
endif

include make/ps.mk

ifndef CXX
CXX = g++
endif

ifndef DEPS_PATH
DEPS_PATH = $(shell pwd)/deps
endif


ifndef PROTOC
PROTOC = ${DEPS_PATH}/bin/protoc
endif

INCPATH = -I./src -I./include -I$(DEPS_PATH)/include -I/opt/intel/mkl/include
CFLAGS = -std=c++11 -msse2 -fPIC -O3 -ggdb -Wall -finline-functions -lmkl_core -lmkl_intel_lp64 -lmkl_intel_thread -liomp5 -lpthread -lm $(INCPATH) $(ADD_CFLAGS)

all: ps test

include make/deps.mk

clean:
	rm -rf build $(TEST) tests/*.d
	find src -name "*.pb.[ch]*" -delete

lint:
	python tests/lint.py ps all include/ps src

ps: build/libps.a

OBJS = $(addprefix build/, customer.o postoffice.o van.o meta.pb.o lr/lr.o lr/util.o ridge_regression/ridge_regression.o)
build/libps.a: $(OBJS)
	ar crv $@ $(filter %.o, $?)

build/%.o: src/%.cc ${ZMQ} src/meta.pb.h
	@mkdir -p $(@D)
	$(CXX) $(INCPATH) -std=c++0x -MM -MT build/$*.o $< >build/$*.d
	$(CXX) $(CFLAGS) -c $< -o $@

src/%.pb.cc src/%.pb.h : src/%.proto ${PROTOBUF}
	$(PROTOC) --cpp_out=./src --proto_path=./src $<

-include build/*.d
-include build/*/*.d

include tests/test.mk
test: $(TEST)

