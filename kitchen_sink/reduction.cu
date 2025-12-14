

#include <stdio.h>
#include <cuda_runtime.h>

// v1: naive
__global__ void reduce_naive(float *in, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        atomicAdd(out, in[idx]);
    }
}

__global__ void reduce_shared(float *in, float *out, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < n) ? in[idx]: 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }

        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(out, sdata[0]);
    }
}

__device__ float warpReduce(float val) {
    for (int offset = 16; offset > 0; offset >>=1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void reduce_warp(float *in, float *out, int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float val = (idx < n) ? in[idx] : 0;
    val = warpReduce(val);

    int lane = tid % 32;
    int warpId = tid / 32;
    if (lane == 0) {
        sdata[warpId] = val;
    }
    __syncthreads();

    if (tid < blockDim.x / 32) {
        val = sdata[tid];
    }
    else {
        val = 0;
    }

    if (warpId == 0) {
        val = warpReduce(val);
        if (tid == 0) {
            atomicAdd(out, val);
        }
    }
}

template <int BLOCK_SIZE>
__global__ void reduce_multi(float *in, float *out, int n) {
    __shared__ float sdata[BLOCK_SIZE / 32];

    int tid = threadIdx.x;
    int idx = blockIdx.x * BLOCK_SIZE * 4 + threadIdx.x;

    float val = 0;
    for (int i = 0; i < 4; i++) {
        int j = idx + i * BLOCK_SIZE;
        if (j >= n) break;
        val += in[j];
    }
    // if (idx < n) val += in[idx];
    // if (idx + BLOCK_SIZE < n) val += in[idx + BLOCK_SIZE];
    // if (idx + BLOCK_SIZE * 2 < n) val += in[idx + BLOCK_SIZE*2];
    // if (idx + BLOCK_SIZE * 3 < n) val += in[idx + BLOCK_SIZE*3];

    val = warpReduce(val);
    int lane = tid % 32;
    int warpId = tid / 32;

    if (lane == 0) sdata[warpId] = val;
    __syncthreads();

    if (tid < BLOCK_SIZE/32) val = sdata[tid];
    else val = 0;

    if (warpId == 0) {
        val = warpReduce(val);
        if (tid == 0) atomicAdd(out, val);
    }
}

void benchmark(const char *name, void (*kernel)(float*, float*, int), 
               float *d_in, float *d_out, int n, int block_size, int smem) {
    cudaMemset(d_out, 0, sizeof(float));
    
    int num_blocks = (n + block_size - 1) / block_size;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    kernel<<<num_blocks, block_size, smem>>>(d_in, d_out, n);
    cudaDeviceSynchronize();
    cudaMemset(d_out, 0, sizeof(float));
    
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        cudaMemset(d_out, 0, sizeof(float));
        kernel<<<num_blocks, block_size, smem>>>(d_in, d_out, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    
    float result;
    cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("%s: %.2f ms avg, result = %.0f, bandwidth = %.2f GB/s\n",
           name, ms / 100, result, (n * sizeof(float) * 100) / (ms * 1e6));
}


void benchmark_multi(const char *name, 
               float *d_in, float *d_out, int n) {
    
    cudaMemset(d_out, 0, sizeof(float));
    const int BLOCK_SIZE = 256;
    int num_blocks = (n + BLOCK_SIZE*4 - 1) / (BLOCK_SIZE*4);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    reduce_multi<BLOCK_SIZE><<<num_blocks, BLOCK_SIZE>>>(d_in, d_out, n);
    cudaDeviceSynchronize();
    cudaMemset(d_out, 0, sizeof(float));
    
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        cudaMemset(d_out, 0, sizeof(float));
        reduce_multi<BLOCK_SIZE><<<num_blocks, BLOCK_SIZE>>>(d_in, d_out, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    
    float result;
    cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("%s: %.2f ms avg, result = %.0f, bandwidth = %.2f GB/s\n",
           name, ms / 100, result, (n * sizeof(float) * 100) / (ms * 1e6));
}

int main() {
    int n = 1 << 24;
    size_t bytes = n * sizeof(float);
    int block_size = 256;

    float *h_in = (float*)malloc(bytes);
    for (int i = 0; i < n; i++) h_in[i] = 1.0f;

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, sizeof(float));
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    printf("Summing %d elements expecting: %d\n\n", n, n);

    benchmark("Naive", reduce_naive, d_in, d_out, n, block_size, 0);
    benchmark("Shared mem", reduce_shared, d_in, d_out, n, block_size, block_size * sizeof(float));
    benchmark("Warp", reduce_warp, d_in, d_out, n, block_size, block_size * sizeof(float));
    benchmark_multi("Warp multi", d_in, d_out, n);

    cudaFree(d_in);
    cudaFree(d_out);

    free(h_in);
    return 0;
    
}