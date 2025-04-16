#include <stdio.h>
#include <stdlib.h>

#define N 3 // Size of the matrix (Change as needed)

// Kernel to process the matrix
__global__ void processMatrix(int *A, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        if (row == col) {
            // Principal diagonal: replace with zero
            A[row * n + col] = 0;
        } else if (row < col) {
            // Above principal diagonal: calculate factorial
            int value = A[row * n + col];
            int factorial = 1;
            for (int i = 1; i <= value; ++i) {
                factorial *= i;
            }
            A[row * n + col] = factorial;
        } else {
            // Below principal diagonal: calculate sum of digits
            int value = A[row * n + col];
            int sum = 0;
            while (value > 0) {
                sum += value % 10;
                value /= 10;
            }
            A[row * n + col] = sum;
        }
    }
}

// Function to print the matrix
void printMatrix(int *A, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%d ", A[i * n + j]);
        }
        printf("\n");
    }
}

int main() {
    int h_A[N * N]; // Host matrix
    int *d_A;
    size_t size = N * N * sizeof(int);

    // Get input matrix from user
    printf("Enter the elements of the %dx%d matrix:\n", N, N);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            scanf("%d", &h_A[i * N + j]);
        }
    }

    // Allocate memory on the device
    cudaMalloc((void **)&d_A, size);

    // Copy the matrix to the device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

    // Launch the kernel
    processMatrix<<<gridDim, blockDim>>>(d_A, N);

    // Copy the processed matrix back to the host
    cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);

    // Print the processed matrix
    printf("Processed Matrix:\n");
    printMatrix(h_A, N);

    // Free device memory
    cudaFree(d_A);

    return 0;
}

