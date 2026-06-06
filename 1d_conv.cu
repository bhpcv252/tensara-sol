/*
 * Problem: https://tensara.org/problems/conv-1d
 * Submission: https://tensara.org/submissions/cmq1br7551mwgwumpxrxq1ut1
 */


#include <cuda_runtime.h>
#define B_MAX 8191

__constant__ float c_B[B_MAX];

__global__ void oDConv(const float* __restrict__ A, float* __restrict__ C, size_t N, size_t K) {

    extern __shared__ float s_A[];


    int ts = blockDim.x + K - 1;

    const int r = (K - 1)/2;
    
    for(int i = threadIdx.x; i < ts; i += blockDim.x) {
        int gi = blockIdx.x * blockDim.x + i - r;
        s_A[i] = (gi >= 0 && gi < N) ? A[gi] : 0.0f;
    }
    
    __syncthreads();

    int tId = threadIdx.x + blockIdx.x * blockDim.x;

    if(tId < N) {
        
        float sum = 0.0f;

        #pragma unroll
        for(int j = 0; j < K; j++){
            sum += s_A[threadIdx.x + j] * c_B[j];
        }

        C[tId] = sum;
    }
}

// Note: A, B, C are device pointers
extern "C" void solution(const float* A, const float* B, float* C, size_t N, size_t K) {
    cudaMemcpyToSymbol(c_B, B, K*sizeof(float));

    int threads = 512;
    int blocks = (N + threads - 1) / threads;
    size_t sB = (threads + K - 1) * sizeof(float);
    oDConv<<<blocks, threads, sB>>>(A, C, N, K);
}
