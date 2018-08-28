USE_GPU := 0

DIR := ./

TF_INC := $(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB := $(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

# NDEBUG fixes a bug: https://github.com/tensorflow/tensorflow/issues/17316
FLAGS := -std=c++11 -shared -fPIC -I$(TF_INC) -I$(TF_INC)/external/nsync/public -L$(TF_LIB) -D_GLIBCXX_USE_CXX11_ABI=0 -O2 -DNDEBUG
CXX := g++
LDFLAGS := -ltensorflow_framework

SOURCES := $(DIR)/ops/*.cpp

ifeq ($(USE_GPU), 1)
    FLAGS += -DGOOGLE_CUDA=1
endif

all: apex_tensorflow.so

apex_tensorflow.so:
	$(CXX) $(FLAGS) $(SOURCES) $(LDFLAGS) -o apex_tensorflow.so

clean:
	rm -rf apex_tensorflow.so

remake: clean all
