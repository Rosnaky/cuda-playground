
#include <vector>
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

using namespace std;

typedef struct {
    double sum;
    double sum_sq;
    int count;
} Stats;

__device__ float sum_warp(float val) {
    for (int i = 16; i > 0; i >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, i);
    }
    return val;
}

__global__ void variance_kernel(const float* __restrict__ input, Stats * __restrict output, int n) {
    extern __shared__ char smem[];

    double *s_sum = (double*) smem;
    double *s_sum_sq = s_sum + blockDim.x;

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    float val = gid < n ? input[gid] : 0;
    val = sum_warp(val);

    int laneId = tid % 32;
    int warpId = tid / 32;
    
    if (laneId == 0) {
        s_sum[warpId] = val;
    }
    __syncthreads();

    if (tid < blockDim.x / 32) {
        val = s_sum[tid];
    }
    else {
        val = 0;
    }

    if (warpId == 0) {
        val = sum_warp(val);
        if (laneId == 0) {
            s_sum[0] = val;
        }
    }

    __syncthreads();

    val = input[gid]*input[gid];

    val = sum_warp(val);
    if (laneId == 0) {
        s_sum_sq[warpId] = val;
    }
    __syncthreads();

    if (tid < blockDim.x / 32) {
        val = s_sum_sq[tid];
    }
    else {
        val = 0;
    }

    if (warpId == 0) {
        val = sum_warp(val);
        if (laneId == 0) {
            s_sum_sq[warpId] = val;
        }
    }

    if (tid == 0) {
        atomicAdd(&output->sum, s_sum[0]);
        atomicAdd(&output->sum_sq, s_sum_sq[0]);
        atomicAdd(&output->count, 1);
    }
}

template<int BLOCK_SIZE, int ELEMENTS_PER_THREAD>
__global__ void variance_kernel_parallel(const float* __restrict__ input, Stats * __restrict output, int n) {
    extern __shared__ char smem[];

    double *s_sum = (double*) smem;
    double *s_sum_sq = s_sum + (BLOCK_SIZE / 32);

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + tid;

    double local_sum = 0;
    double local_sum_sq = 0;
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        int idx = gid + i * BLOCK_SIZE;
        if (idx < n) {
            double val = input[idx];
            local_sum += val;
            local_sum_sq += val * val;
        }
    }

    local_sum = sum_warp(local_sum);
    local_sum_sq = sum_warp(local_sum_sq);

    int laneId = tid % 32;
    int warpId = tid / 32;
    
    if (laneId == 0) {
        s_sum[warpId] = local_sum;
        s_sum_sq[warpId] = local_sum_sq;
    }
    __syncthreads();

    if (tid < blockDim.x / 32) {
        local_sum = s_sum[tid];
        local_sum_sq = s_sum_sq[tid];
    }
    else {
        local_sum = 0;
        local_sum_sq = 0;
    }

    if (warpId == 0) {
        local_sum = sum_warp(local_sum);
        local_sum_sq = sum_warp(local_sum_sq);
        if (laneId == 0) {
            atomicAdd(&output->sum, local_sum);
            atomicAdd(&output->sum_sq, local_sum_sq);
            atomicAdd(&output->count, 1);
        }
    }
}

int main() {
    const int N = 1 << 24;
    const int BLOCK_SIZE = 256;
    const int ELEMENTS_PER_THREAD = 8;
    const int NUM_BLOCKS = (N + BLOCK_SIZE * ELEMENTS_PER_THREAD - 1) / 
                           (BLOCK_SIZE * ELEMENTS_PER_THREAD);  // FIXED

    vector<float> h_data(N);
    double sum = 0, sum_sq = 0;

    for (int i = 0; i < N; i++) {
        h_data[i] = (i % 1000) / 100.0f;
        sum += h_data[i];
        sum_sq += h_data[i] * h_data[i];
    }

    double mean = sum / N;
    double variance = (sum_sq / N) - mean * mean;

    printf("N = %d elements\n", N);
    printf("True Mean = %.6f\n", mean);
    printf("True Variance = %.6f\n", variance);
    printf("True Stddev = %.6f\n\n", sqrt(variance));

    float* d_data;
    Stats* d_stats;
    size_t smem_size = 2 * (BLOCK_SIZE / 32) * sizeof(double);
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMalloc(&d_stats, sizeof(Stats));
    cudaMemcpy(d_data, h_data.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemset(d_stats, 0, sizeof(Stats));
    variance_kernel_parallel<BLOCK_SIZE, ELEMENTS_PER_THREAD><<<NUM_BLOCKS, BLOCK_SIZE, smem_size>>>(d_data, d_stats, N);
    cudaDeviceSynchronize();

    const int NUM_TRIALS = 100;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < NUM_TRIALS; i++) {
        cudaMemset(d_stats, 0, sizeof(Stats));
        variance_kernel_parallel<BLOCK_SIZE, ELEMENTS_PER_THREAD><<<NUM_BLOCKS, BLOCK_SIZE, smem_size>>>(d_data, d_stats, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms;
    cudaEventElapsedTime(&total_ms, start, stop);
    float avg_ms = total_ms / NUM_TRIALS;

    Stats h_stats;
    cudaMemcpy(&h_stats, d_stats, sizeof(Stats), cudaMemcpyDeviceToHost);

    double gpu_mean = h_stats.sum / N;
    double gpu_var = (h_stats.sum_sq / N) - (gpu_mean * gpu_mean);

    printf("GPU Results:\n");
    printf("Mean: %.6f (error: %e)\n", gpu_mean, fabs(gpu_mean - mean));
    printf("Variance: %.6f (error: %e)\n", gpu_var, fabs(gpu_var - variance));
    printf("Stddev: %.6f\n\n", sqrt(gpu_var));

    double bytes = N * sizeof(float);
    double bandwidth = bytes / (avg_ms * 1e6);
    printf("Time: %.3f ms\n", avg_ms);
    printf("Bandwidth: %.1f GB/s\n", bandwidth);

    cudaFree(d_data);
    cudaFree(d_stats);

    return 0;
}