CC = g++
NVCC = nvcc

MNIST_ML_ROOT := $(realpath $(CURDIR)/..)
SRC := $(MNIST_ML_ROOT)/KNN/src
INCLUDE_DIR := $(MNIST_ML_ROOT)/KNN/include
LIB_DIR := $(MNIST_ML_ROOT)/lib

CFLAGS := -std=c++17 -DEUCLID
NVCCFLAGS := -std=c++17 -arch=sm_75 --compiler-bindir /usr/bin/g++-10
LDFLAGS := -L$(LIB_DIR) -ldata
INCLUDES := -I$(INCLUDE_DIR) -I$(MNIST_ML_ROOT)/include

all: main_cuda

# Step 1: build GPU object
obj/knn_cuda.o: $(SRC)/knn.cu
	mkdir -p obj
	$(NVCC) $(NVCCFLAGS) -c $< -o $@ $(INCLUDES)

# Step 2: build CPU + link GPU
main_cuda: obj/knn_cuda.o $(SRC)/knn.cpp
	$(CC) $(CFLAGS) $(SRC)/knn.cpp obj/knn_cuda.o -o $@ $(INCLUDES) $(LDFLAGS) -lcudart -Xlinker -rpath=$(LIB_DIR)

clean:
	rm -rf main_cuda obj/
