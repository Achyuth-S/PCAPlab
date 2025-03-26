#include <stdio.h>
#include <cuda.h>
#include <math.h>

__global__ void computeSine(float *angles, float *results, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) results[i] = sinf(angles[i]);
}

int main() {
    int N = 5;
    float angles[N] = {0, 1.57, 3.14, 4.71, 6.28}, results[N];
    float *d_angles, *d_results;
    
    cudaMalloc(&d_angles, N * sizeof(float));
    cudaMalloc(&d_results, N * sizeof(float));
    
    cudaMemcpy(d_angles, angles, N * sizeof(float), cudaMemcpyHostToDevice);

    computeSine<<<1, N>>>(d_angles, d_results, N);
    
    cudaMemcpy(results, d_results, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < N; i++)
        printf("%f ", results[i]);

    cudaFree(d_angles); cudaFree(d_results);
    return 0;
}