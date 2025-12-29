
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


__global__ void variance_kernel_parallel(const float* __restrict__ input, Stats * __restrict output, int n) {
    extern __shared__ char smem[];

    double *s_sum = (double*) smem;
    double *s_sum_sq = s_sum + blockDim.x;

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    float val = gid < n ? input[gid] : 0;
    float val1 = val * val;
    val = sum_warp(val);
    val1 = sum_warp(val1);

    int laneId = tid % 32;
    int warpId = tid / 32;
    
    if (laneId == 0) {
        s_sum[warpId] = val;
        s_sum_sq[warpId] = val1;
    }
    __syncthreads();

    if (tid < blockDim.x / 32) {
        val = s_sum[tid];
        val1 = s_sum_sq[tid];
    }
    else {
        val = 0;
        val1 = 0;
    }

    if (warpId == 0) {
        val = sum_warp(val);
        val1 = sum_warp(val1);
        if (laneId == 0) {
            atomicAdd(&output->sum, val);
            atomicAdd(&output->sum_sq, val1);
            atomicAdd(&output->count, 1);
        }
    }
}

int main() {
    const int N = 1 << 24;
    const int BLOCK_SIZE = 256;
    const int NUM_BLOCKS = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;


    vector<float> h_data(N);
    double sum = 0, sum_sq = 0;

    for (int i = 0; i < N; i++) {
        h_data[i] = (i%1000)/100.0;
        sum += h_data[i];
        sum_sq += h_data[i] * h_data[i];
    }

    double mean = sum / N;
    double variance = (sum_sq/N) - mean * mean;

    printf("N = %d elements\n", N);
    printf("True Mean = %.6f\n", mean);
    printf("True Variance = %.6f\n", variance);
    printf("True stddev = %.6f\n", sqrt(variance));

    const int NUM_TRIALS = 100;
    Stats h_stats;
    double avg_ms = 0;
    
    for (int i = 0; i < NUM_TRIALS; i++) {
        float * d_data;
        Stats * d_stats;
    
        cudaMalloc(&d_data, N*sizeof(float));
        cudaMalloc(&d_stats, sizeof(Stats));
    
        cudaMemcpy(d_data, h_data.data(), N*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_stats, 0, sizeof(Stats));
    
        size_t smem_size = 2 * BLOCK_SIZE * sizeof(double);
    

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    
        cudaEventRecord(start);
        variance_kernel_parallel<<<NUM_BLOCKS, BLOCK_SIZE, smem_size>>>(d_data, d_stats, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
    
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        avg_ms += ms;
    
        cudaMemcpy(&h_stats, d_stats, sizeof(Stats), cudaMemcpyDeviceToHost);
        
        cudaFree(d_data);
        cudaFree(d_stats);
    }
    avg_ms /= NUM_TRIALS;
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
    

}