
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <random>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
cudaError_t scanWithCuda(int* a, const int* b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

// call on 512 threads
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
    // sync global memory
    threadfence_block();
    //collect previous sums to have full cumulative value  
    // zero out block memory
    s[li] = 0;
    // copy previous sum values to shared
    if (li < blockIdx.x)
        s[li] = cumsum[blockDim.x * li + 1023];
    __syncthreads();

    // mathematical indexing
    auto ni = li + 1;
    // upsweep
#pragma unroll
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
        cumsum[i] += s[1023];
    __syncthreads();
}

int main()
{
    const int arraySize = 10;
    int a[arraySize];
    int b[arraySize];

    std::random_device rd;
    auto gen = std::mt19937(rd());
    auto distribution = std::binomial_distribution<int>(1023, 1. / 128.);

    for (auto& i : b)
        i = distribution(gen);

    // Add vectors in parallel.
    cudaError_t cudaStatus = scanWithCuda(a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    for (auto i : b)
        std::cout << i << ',';
    std::cout<<std::endl;
    for (auto i = 10; i-- > 0;)
        std::cout << '-';
    std::cout << std::endl;
    for (auto i : a)
        std::cout << i << ',';

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
cudaError_t scanWithCuda(int* a, const int* b, unsigned int size)
{
    int* dev_a = 0;
    int* dev_b = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
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
    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    // Launch a kernel on the GPU with one thread for each element.
    scanKernel <<<4, size >> > (dev_a, dev_b);

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
    cudaStatus = cudaMemcpy(a, dev_a, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_a);
    cudaFree(dev_b);

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
