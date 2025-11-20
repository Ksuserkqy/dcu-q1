// hip.cpp - Q1 (DCU)
// by Ksuserkqy(20251113620)
// Docs: https://www.ksuser.cn/dcu/
// 2025-10-14

#include <iostream>
#include <hip/hip_runtime.h>
using namespace std;

static constexpr int N = 10000;

__global__ void init_kernel(float* __restrict__ A, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * N;
    if (idx < total) {
        unsigned int s = seed ^ idx;
        s = s * 1103515245 + 12345;
        A[idx] = (s & 0x7FFFFFFF) / (float)0x7FFFFFFF;
    }
}

// 每个 block 对应一行，使用 block 内并行归约对该行求和。
__global__ void sum_rows_kernel(float* __restrict__ A, float* __restrict__ rowSums) {
    int row = blockIdx.x;
    if (row >= N) return;

    // 每个线程对该行的部分列求和（按 blockDim.x 跨列跳步）
    float local = 0.0f;
    for (int col = threadIdx.x; col < N; col += blockDim.x) {
        local += A[row * N + col];
    }

    // 在共享内存中进行归约 (block)
    extern __shared__ float sdata[];
    sdata[threadIdx.x] = local;
    __syncthreads();

    // 基于树形的归约（假设 blockDim.x 为 2 的幂，或者通过边界检查处理奇数情况）
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            sdata[threadIdx.x] += sdata[threadIdx.x + offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        rowSums[row] = sdata[0];
        A[row * N] = sdata[0];
    }
}

int main() {
    // 设备缓冲区
    float* A_dev = nullptr;
    float* rowSums_dev = nullptr;
    size_t bytesA = static_cast<size_t>(N) * N * sizeof(float);
    size_t bytesRow = static_cast<size_t>(N) * sizeof(float);
    hipMalloc(&A_dev, bytesA);
    hipMalloc(&rowSums_dev, bytesRow);

    // Kernel 启动配置
    dim3 blockInit(256);
    dim3 gridInit((N * N + blockInit.x - 1) / blockInit.x);
    dim3 blockSum(256);
    dim3 gridSum(N); // one block per row

    // 用于精确测量 GPU 时间的事件
    hipEvent_t eStart, eStop;
    hipEventCreate(&eStart);
    hipEventCreate(&eStop);

    // 初始化计时
    hipEventRecord(eStart, 0);
    hipLaunchKernelGGL(init_kernel, gridInit, blockInit, 0, 0, A_dev, 14);
    hipEventRecord(eStop, 0);
    hipEventSynchronize(eStop);
    float msInit = 0.0f;
    hipEventElapsedTime(&msInit, eStart, eStop);
    cout << "初始化耗时: " << (msInit / 1000.0f) << " 秒" << endl;

    // 求和计时（需要共享内存大小 = blockSum.x * sizeof(float)）
    hipEventRecord(eStart, 0);
    hipLaunchKernelGGL(sum_rows_kernel, gridSum, blockSum, blockSum.x * sizeof(float), 0, A_dev, rowSums_dev);
    hipEventRecord(eStop, 0);
    hipEventSynchronize(eStop);
    float msSum = 0.0f;
    hipEventElapsedTime(&msSum, eStart, eStop);
    cout << "累加计算耗时: " << (msSum / 1000.0f) << " 秒" << endl;

    // 拷回前 5 行的累加结果用于验证（rowSums）以及在 A 中对应行首的实际值
    float rowSums_host[5];
    hipMemcpy(rowSums_host, rowSums_dev, 5 * sizeof(float), hipMemcpyDeviceToHost);

    float rowHeads_host[5];
    hipMemcpy2D(rowHeads_host, sizeof(float), A_dev, N * sizeof(float), sizeof(float), 5, hipMemcpyDeviceToHost);

    for (int i = 0; i < 5; i++) {
        cout << "第 " << i << " 行的累加值 = " << rowHeads_host[i]
             << " (rowSums: " << rowSums_host[i] << ")" << endl;
    }

    // 清理资源
    hipEventDestroy(eStart);
    hipEventDestroy(eStop);
    hipFree(rowSums_dev);
    hipFree(A_dev);
}
