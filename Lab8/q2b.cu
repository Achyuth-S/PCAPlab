#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

// Kernel for column-based computation
__global__ void ColBasedMatrixMul(int* A, int* B, int* C, int n) {
    int col = blockIdx.x; // Each thread block computes one column
    if (col < n) {
        for (int row = 0; row < n; row++) {
            int sum = 0;
            for (int k = 0; k < n; k++) {
                sum += A[row * n + k] * B[k * n + col];
            }
            C[row * n + col] = sum;
        }
    }
}

int main() {
    int n;
    printf("Enter dimension of matrix(nxn): ");
    scanf("%d", &n);

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

    int *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, n * n * sizeof(int));
    cudaMalloc((void**)&d_B, n * n * sizeof(int));
    cudaMalloc((void**)&d_C, n * n * sizeof(int));

    cudaMemcpy(d_A, h_A, n * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n * n * sizeof(int), cudaMemcpyHostToDevice);

    ColBasedMatrixMul<<<n, 1>>>(d_A, d_B, d_C, n);

    cudaMemcpy(h_C, d_C, n * n * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Resultant Matrix using column based:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%d ", h_C[i * n + j]);
        }
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}

