#include <stdio.h>
#define MASK_WIDTH 3
#define TILE_WIDTH 8

__constant__ int d_M[MASK_WIDTH]; // Constant memory for mask

__global__ void convolution1D(int *N, int *P, int width) {
    __shared__ int Ns[TILE_WIDTH + MASK_WIDTH - 1];

    int tx = threadIdx.x;
    int row_o = blockIdx.x * TILE_WIDTH + tx;
    int row_i = row_o - (MASK_WIDTH / 2);

    // Load data into shared memory
    if (row_i >= 0 && row_i < width) {
        Ns[tx] = N[row_i];
    } else {
        Ns[tx] = 0;
    }

    __syncthreads();

    // Perform convolution
    int result = 0;
    if (tx < TILE_WIDTH && row_o < width) {
        for (int i = 0; i < MASK_WIDTH; i++) {
            result += d_M[i] * Ns[tx + i];
        }
        P[row_o] = result;
    }
}

int main() {
    const int width = 16;
    const int mask_width = MASK_WIDTH;
    int h_N[width], h_P[width];
    int h_M[MASK_WIDTH] = {1, 0, -1}; // Example: simple edge detection mask

    for (int i = 0; i < width; i++) h_N[i] = i + 1;

    int *d_N, *d_P;
    cudaMalloc(&d_N, width * sizeof(int));
    cudaMalloc(&d_P, width * sizeof(int));

    cudaMemcpy(d_N, h_N, width * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_M, h_M, MASK_WIDTH * sizeof(int)); // copy mask to constant memory

    int blocks = (width + TILE_WIDTH - 1) / TILE_WIDTH;
    convolution1D<<<blocks, TILE_WIDTH + MASK_WIDTH - 1>>>(d_N, d_P, width);

    cudaMemcpy(h_P, d_P, width * sizeof(int), cudaMemcpyDeviceToHost);

    // Print output
    for (int i = 0; i < width; i++) {
        printf("P[%d] = %d\n", i, h_P[i]);
    }

    cudaFree(d_N);
    cudaFree(d_P);
    return 0;
}
