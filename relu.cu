/*
 * Problem: https://tensara.org/problems/relu
 * Submission: https://tensara.org/submissions/cmq22yp4309m71159qyhl0yhj
 */


#include <cuda_runtime.h>

__global__ void relu4(const float4* __restrict__ i, float4* __restrict__ o, size_t nm4) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if(tid < nm4) {
        float4 a = i[tid]; 
        o[tid] = make_float4(
            max(0.0f, a.x),
            max(0.0f, a.y),
            max(0.0f, a.z),
            max(0.0f, a.w)
        );
    }
}

__global__ void relu(const float* __restrict__ i, float* __restrict__ o, size_t start, size_t nm) {
    int tid = threadIdx.x + start;

    if(tid < nm) {
        o[tid] = max(0.0f, i[tid]);
    }
}

extern "C" void solution(const float* input, float* output, size_t n, size_t m) {

    size_t nm = n*m;
    size_t nm4 = nm / 4;

    if(nm4) {
        int threads = 256;
        int blocks = ((nm4) + threads - 1) / threads;
    
        relu4<<<blocks, threads>>>(
            reinterpret_cast<const float4*>(input),
            reinterpret_cast<float4*>(output),
            nm4
        );
    }

    int rem = static_cast<int>(nm % 4);

    if(rem) {
        size_t start = nm4 * 4;
        relu<<<1, rem>>>(
            input,
            output,
            start,
            nm
        );
    }



}
