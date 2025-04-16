#include <stdio.h>
#include <cuda.h>

// CUDA kernel to process the matrix
__global__ void processMatrix(int *A, int *B, int *rowSum, int *colSum, int rows, int cols) {
    int row = blockIdx.x;
    int col = threadIdx.x;

    if (row < rows && col < cols) {
        int val = A[row * cols + col];
        if (val % 2 == 0) {
            B[row * cols + col] = rowSum[row];
        } else {
            B[row * cols + col] = colSum[col];
        }
    }
}

int main() {
    int rows, cols;

    // Input the dimensions of the matrix
    printf("Enter the number of rows: ");
    scanf("%d", &rows);
    printf("Enter the number of columns: ");
    scanf("%d", &cols);

    int *A = (int *)malloc(rows * cols * sizeof(int));
    int *B = (int *)malloc(rows * cols * sizeof(int));
    int *rowSum = (int *)malloc(rows * sizeof(int));
    int *colSum = (int *)malloc(cols * sizeof(int));

    // Initialize row and column sums to zero
    for (int i = 0; i < rows; i++) rowSum[i] = 0;
    for (int j = 0; j < cols; j++) colSum[j] = 0;

    // Input matrix elements
    printf("Enter the elements of the matrix:\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            scanf("%d", &A[i * cols + j]);
            rowSum[i] += A[i * cols + j];
            colSum[j] += A[i * cols + j];
        }
    }

    // Allocate memory on the GPU
    int *d_A, *d_B, *d_rowSum, *d_colSum;
    cudaMalloc((void **)&d_A, rows * cols * sizeof(int));
    cudaMalloc((void **)&d_B, rows * cols * sizeof(int));
    cudaMalloc((void **)&d_rowSum, rows * sizeof(int));
    cudaMalloc((void **)&d_colSum, cols * sizeof(int));

    // Copy data to the GPU
    cudaMemcpy(d_A, A, rows * cols * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rowSum, rowSum, rows * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colSum, colSum, cols * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    processMatrix<<<rows, cols>>>(d_A, d_B, d_rowSum, d_colSum, rows, cols);

    // Copy the result back to the CPU
    cudaMemcpy(B, d_B, rows * cols * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the resultant matrix
    printf("Resultant Matrix B:\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", B[i * cols + j]);
        }
        printf("\n");
    }

    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_rowSum);
    cudaFree(d_colSum);

    // Free CPU memory
    free(A);
    free(B);
    free(rowSum);
    free(colSum);

    return 0;
}

