/*
 * Problem: https://tensara.org/problems/vector-addition
 * Submission: https://tensara.org/submissions/cmpw9ip450i9ewumpbf6wtpm6
 */


#include <cuda_runtime.h>

__global__ void va4(const float4* i1, const float4* i2, float4* o, size_t n4) {
    int tId = threadIdx.x + blockIdx.x * blockDim.x;

    if(tId < n4) {
        float4 a = i1[tId];
        float4 b = i2[tId];
        o[tId] = make_float4(
            a.x + b.x,
            a.y + b.y,
            a.z + b.z,
            a.w + b.w
        );
    }
}

__global__ void va(const float* i1, const float* i2, float* o, size_t start, size_t n) {
    int tId = start + threadIdx.x;
    if(tId < n) {
        o[tId] = i1[tId] + i2[tId];
    }
}

extern "C" void solution(const float* d_input1, const float* d_input2, float* d_output, size_t n) {

    size_t n4 = n / 4;

    if(n4) {
        int threads = 256;
        int blocks = (n4 + threads - 1) / threads;

        va4<<<blocks, threads>>>(
            reinterpret_cast<const float4*>(d_input1),  
            reinterpret_cast<const float4*>(d_input2),  
            reinterpret_cast<float4*>(d_output),  
            n4
        );
    }

    int rem = static_cast<int>(n % 4);
    size_t start = n4 * 4;

    if(rem) {
        va<<<1, rem>>>(d_input1, d_input2, d_output, start, n);
    }


}
