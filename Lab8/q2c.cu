#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

// Kernel for element-based computation
__global__ void ElementBasedMatrixMul(int* A, int* B, int* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Compute row index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Compute column index

    if (row < n && col < n) {
        int sum = 0;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

int main() {
    int n;
    printf("Enter dimension of matrix(nxn): ");
    scanf("%d", &n);

    // Allocate host memory
    int *h_A = (int*)malloc(n * n * sizeof(int));
    int *h_B = (int*)malloc(n * n * sizeof(int));
    int *h_C = (int*)malloc(n * n * sizeof(int));

    // Initialize matrices
    printf("Enter elements of matrix A:\n");
    for (int i = 0; i < n * n; i++) {
        scanf("%d", &h_A[i]);
    }

    printf("Enter elements of matrix B:\n");
    for (int i = 0; i < n * n; i++) {
        scanf("%d", &h_B[i]);
    }

    // Device memory allocation
    int *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, n * n * sizeof(int));
    cudaMalloc((void**)&d_B, n * n * sizeof(int));
    cudaMalloc((void**)&d_C, n * n * sizeof(int));

    // Copy matrices from host to device
    cudaMemcpy(d_A, h_A, n * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n * n * sizeof(int), cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockDim(16, 16); // 16 x 16 threads per block
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);

    // Launch kernel
    ElementBasedMatrixMul<<<gridDim, blockDim>>>(d_A, d_B, d_C, n);

    // Copy result matrix back to host
    cudaMemcpy(h_C, d_C, n * n * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the result matrix
    printf("Resultant Matrix using element based:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%d ", h_C[i * n + j]);
        }
        printf("\n");
    }

    // Free device and host memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}

