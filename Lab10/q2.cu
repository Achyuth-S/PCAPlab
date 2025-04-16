#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <cuda.h>
#define N 8
#define MASK_WIDTH 3

__constant__ int d_mask[MASK_WIDTH];

__global__ void conv1D(int *input, int *output) {
    int i = threadIdx.x;
    int val = 0;
    for (int j = 0; j < MASK_WIDTH; j++) {
        int idx = i + j - 1;
        if (idx >= 0 && idx < N)
            val += input[idx] * d_mask[j];
    }
    output[i] = val;
}

int main() {
    int h_input[N] = {1, 2, 3, 4, 5, 6, 7, 8};
    int h_output[N];
    int h_mask[MASK_WIDTH] = {1, 0, -1};

    int *d_input, *d_output;
    cudaMalloc(&d_input, sizeof(int) * N);
    cudaMalloc(&d_output, sizeof(int) * N);

    cudaMemcpy(d_input, h_input, sizeof(int) * N, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_mask, h_mask, sizeof(int) * MASK_WIDTH);

    conv1D<<<1, N>>>(d_input, d_output);

    cudaMemcpy(h_output, d_output, sizeof(int) * N, cudaMemcpyDeviceToHost);

    printf("Conv Output Array:\n");
    for (int i = 0; i < N; i++) {
        printf("%d ", h_output[i]);
        printf("\n");
    }


    cudaFree(d_input); 
    cudaFree(d_output);
    return 0;
}
