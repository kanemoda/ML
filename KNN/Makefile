CC=g++
MNIST_ML_ROOT := $(realpath $(CURDIR)/..)
SRC := $(MNIST_ML_ROOT)/KNN/src
CFLAGS := -std=c++11 -DEUCLID
INCLUDE_DIR := $(MNIST_ML_ROOT)/KNN/include

all: main

main : $(SRC)/knn.cpp
	$(CC) $(CFLAGS) $(SRC)/knn.cpp -o main -L$(MNIST_ML_ROOT)/lib/ -I$(INCLUDE_DIR) -I$(MNIST_ML_ROOT)/include -ldata -Wl,-rpath,$(MNIST_ML_ROOT)/lib


clean:
	rm main