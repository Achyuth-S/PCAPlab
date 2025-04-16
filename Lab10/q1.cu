#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <cuda.h>
#define N 2

__global__ void matMul(int *a, int *b, int *c) {
    int row = threadIdx.y;
    int col = threadIdx.x;
    int sum = 0;
    for (int k = 0; k < N; k++)
        sum += a[row * N + k] * b[k * N + col];
    c[row * N + col] = sum;
}

int main() {
    int a[N*N] = {1, 2, 3, 4};
    int b[N*N] = {5, 6, 7, 8};
    int c[N*N];

    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, sizeof(int)*N*N);
    cudaMalloc(&d_b, sizeof(int)*N*N);
    cudaMalloc(&d_c, sizeof(int)*N*N);

    cudaMemcpy(d_a, a, sizeof(int)*N*N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(int)*N*N, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(N, N);
    matMul<<<1, threadsPerBlock>>>(d_a, d_b, d_c);

    cudaMemcpy(c, d_c, sizeof(int)*N*N, cudaMemcpyDeviceToHost);

   printf("Resultant Matrix:\n");
    for (int i = 0; i < N*N; i++) {
        printf("%d ", c[i]);
        if ((i + 1) % N == 0) std::cout << "\n";
    }

    cudaFree(d_a); 
    cudaFree(d_b); 
    cudaFree(d_c);
    return 0;
}
