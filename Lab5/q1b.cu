#include <stdio.h>
#include <cuda.h>

__global__ void addVectors(float *A, float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

int main() {
    int N = 5;
    float A[N] = {1, 2, 3, 4, 5}, B[N] = {10, 20, 30, 40, 50}, C[N];
    float *d_A, *d_B, *d_C;
    
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));
    
    cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice);

    addVectors<<<1, N>>>(d_A, d_B, d_C, N);
    
    cudaMemcpy(C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < N; i++)
        printf("%f ", C[i]);
    
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}