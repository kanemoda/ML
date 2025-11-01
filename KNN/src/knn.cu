#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <stdint.h>
#include "data_handler.hpp"
#include "knn.hpp"

#define MAX_FEATURES 784 // MNIST 28x28 için


// CONSTANT MEMORY (QUERY)

__constant__ uint8_t d_query_const[MAX_FEATURES];


// CUDA Kernel

__global__ void compute_distances_kernel(
    const uint8_t *train,
    float *distances,
    int feature_size,
    int num_train)
{
    extern __shared__ uint8_t query_shared[];  // Dinamik shared memory

    // 🔹 Query’yi shared memory’ye kopyala
    for (int i = threadIdx.x; i < feature_size; i += blockDim.x)
        query_shared[i] = d_query_const[i];
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_train)
    {
        float sum = 0.0f;
        // 🔹 Artık query_shared kullanıyoruz
        for (int i = 0; i < feature_size; ++i)
        {
            float diff = (float)query_shared[i] - (float)train[idx * feature_size + i];
            sum += diff * diff;
        }
        distances[idx] = sqrtf(sum);
    }
}

// Host-side function

void knn::find_k_nearest(data *query_point)
{
    neighbors = new std::vector<data *>;
    neighbors->reserve(k);

    int num_train = training_data->size();
    int feature_size = query_point->get_feature_vector_size();

    // 1️⃣ Flatten training data
    std::vector<uint8_t> train_flat(num_train * feature_size);
    for (int i = 0; i < num_train; ++i)
    {
        const auto &vec = *training_data->at(i)->get_feature_vector();
        std::copy(vec.begin(), vec.end(), train_flat.begin() + i * feature_size);
    }

    std::vector<float> distances(num_train);

    // 2️⃣ GPU memory allocate
    uint8_t *d_train = nullptr;
    float *d_distances = nullptr;
    cudaMalloc(&d_train, num_train * feature_size * sizeof(uint8_t));
    cudaMalloc(&d_distances, num_train * sizeof(float));

    // 3️⃣ Query’yi constant memory’ye yükle
    const std::vector<uint8_t> &query = *query_point->get_feature_vector();
    cudaMemcpyToSymbol(d_query_const, query.data(), feature_size * sizeof(uint8_t));

    // 4️⃣ Training verisini kopyala
    cudaMemcpy(d_train, train_flat.data(), num_train * feature_size * sizeof(uint8_t), cudaMemcpyHostToDevice);

    // 5️⃣ CUDA zaman ölçümü başlat
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // 6️⃣ Kernel launch
    int threads = 256;
    int blocks = (num_train + threads - 1) / threads;
    size_t shared_mem_size = feature_size * sizeof(uint8_t);
    compute_distances_kernel<<<blocks, threads, shared_mem_size>>>(
        d_train, d_distances, feature_size, num_train);

    // 7️⃣ Zaman ölçümü bitir
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("KNN CUDA Kernel Time: %.3f ms\n", milliseconds);

    // 8️⃣ Devam (veriyi host’a al)
    cudaMemcpy(distances.data(), d_distances, num_train * sizeof(float), cudaMemcpyDeviceToHost);

    // 9️⃣ K en yakın seç
    std::vector<std::pair<float, int>> dist_idx(num_train);
    for (int i = 0; i < num_train; ++i)
        dist_idx[i] = {distances[i], i};

    size_t limit = std::min((size_t)k, dist_idx.size());
    std::nth_element(dist_idx.begin(), dist_idx.begin() + limit, dist_idx.end(),
                     [](auto &a, auto &b) { return a.first < b.first; });

    for (size_t i = 0; i < limit; ++i)
        neighbors->push_back(training_data->at(dist_idx[i].second));

    // 🔹 Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_train);
    cudaFree(d_distances);
}

