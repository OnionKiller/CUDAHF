﻿
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <random>
#include <algorithm>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
cudaError_t scanWithCuda(int * ret, const int * in, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}




//could be a call on 512 threads
__global__ void scanKernel(int* cumsum, int* data) {
    auto i = threadIdx.x + blockDim.x * blockIdx.x;
    auto li = threadIdx.x;
    if(li > 1024)
        return;
    // move to shared memory per block
    __shared__ int s[1024];
    s[li] = data[i];

    __syncthreads();

    // mathematical indexing
    auto ni = li + 1;
    // upsweep
    #pragma unroll
    for (auto t = 1; t <= 10; t++)
    {
        auto shift = 1 << t-1;
        if (ni % (1 << t) == 0)
        {
            s[ni - 1] += s[ni - shift - 1];
        }
        __syncthreads();
    }


    // downsweep
    #pragma unroll
    for (auto t = 10; t > 0; t--)
    {
        auto shift = 1 << t - 1;
        // last index when the addition is not possible (it is known to be the the last index only affected)
        if (ni != 1024 && ni % (1 << t) == 0)
        {
            s[ni + shift - 1] += s[ni - 1];
        }
        __syncthreads();
    }

    cumsum[i] = s[li];
    __syncthreads();
}

//collect previous sums to have full cumulative value  
__global__ void scanPartialResults(int* sum,int* data) {
    auto i = threadIdx.x + blockDim.x * blockIdx.x;
    auto li = threadIdx.x;
    auto ni = li + 1;

    __shared__ int s[1024];
    s[li] = 0;
    // copy previous sum values to shared
    if (li < blockIdx.x)
        s[li] = data[blockDim.x * li + 1023];
    __syncthreads();

    // upsweep
    //#pragma unroll
    for (auto t = 1; t <= 10; t++)
    {
        auto shift = 1 << t - 1;
        if (ni % (1 << t) == 0)
        {
            s[ni - 1] += s[ni - shift - 1];
        }
        __syncthreads();
    }

    //add cumulative sum
    sum[i] = data[i] + s[1023];
    __syncthreads();
}

int main()
{
    const int arraySize = 1024*1024;
    int* a = new int[arraySize];
    int* b = new int[arraySize];

    std::random_device rd;
    auto gen = std::mt19937(rd());
    auto distribution = std::binomial_distribution<int>(1023, 1. / 128.);

    std::generate(a, a + arraySize, [&]() {
        return 1;
        });

    // Add vectors in parallel.
    cudaError_t cudaStatus = scanWithCuda(b, a, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    std::cout << "test full sum" << std::endl;
    for (auto i = 2025; i < 2050; i++)
        std::cout << a[i] << ',';
    std::cout<<std::endl;
    for (auto i = 10; i-- > 0;)
        std::cout << '-';
    std::cout << std::endl;
    for (auto j = 1; j < 1024; j++)
    {
        if(b[j*1024]-b[j*1024-1] != 1)
        {
            std::cout << std::endl << "error at:" << j << std::endl;
        for(auto i = j*1024-5;i<j*1024+5;i++)
            std::cout << b[i] << ',';
        }
    }
    std::cout << std::endl << " Full sum is: " << b[1024 * 1024 - 1] << " expected: " << 1024 * 1024;

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t scanWithCuda(int * ret, const int * in, unsigned int size)
{
    int* dev_ret = 0;
    int* dev_in = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_ret, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_in, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_in, in, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    // Launch a kernel on the GPU with one thread for each element.
    scanKernel <<<1024, 1024 >> > (dev_ret, dev_in);
    // sync blocks
    // sum over blocks
    scanPartialResults<<<1024, 1024>>>(dev_in, dev_ret);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(ret, dev_in, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_ret);
    cudaFree(dev_in);

    return cudaStatus;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
