#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <cuda.h>
#define N 8

__global__ void inclusiveScan(int *a, int *b) {
    int i = threadIdx.x;
    b[i] = a[i];
    __syncthreads();

    for (int offset = 1; offset < N; offset *= 2) {
        int temp = 0;
        if (i >= offset) temp = b[i - offset];
        __syncthreads();
        b[i] += temp;
        __syncthreads();
    }
}

int main() {
    int h_a[N] = {1, 2, 3, 4, 5, 6, 7, 8};
    int h_b[N];

    int *d_a, *d_b;
    cudaMalloc(&d_a, sizeof(int) * N);
    cudaMalloc(&d_b, sizeof(int) * N);

    cudaMemcpy(d_a, h_a, sizeof(int) * N, cudaMemcpyHostToDevice);
    inclusiveScan<<<1, N>>>(d_a, d_b);
    cudaMemcpy(h_b, d_b, sizeof(int) * N, cudaMemcpyDeviceToHost);

    printf("Inclusive Scan Output Array:\n");
    for (int i = 0; i < N; i++) printf("%d ", h_b[i]);
    printf("\n");

    cudaFree(d_a); 
    cudaFree(d_b);
    return 0;
}
