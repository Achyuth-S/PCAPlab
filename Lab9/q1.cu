// Sparse Matrix: 3x3
// A =  10  0  0
//       0 20  0
//       0  0 30

#include <stdio.h>

__global__ void spmv_csr(int *values, int *rowPtr, int *colInd, int *x, int *y, int N) {
    int row = threadIdx.x;
    int dot = 0;
    for (int i = rowPtr[row]; i < rowPtr[row + 1]; i++) {
        dot += values[i] * x[colInd[i]];
    }
    y[row] = dot;
}

int main() {
    const int N = 3;
    int h_values[] = {10, 20, 30};
    int h_colInd[] = {0, 1, 2};
    int h_rowPtr[] = {0, 1, 2, 3};
    int h_x[] = {1, 2, 3};
    int h_y[N];

    int *d_values, *d_colInd, *d_rowPtr, *d_x, *d_y;
    cudaMalloc(&d_values, 3 * sizeof(int));
    cudaMalloc(&d_colInd, 3 * sizeof(int));
    cudaMalloc(&d_rowPtr, 4 * sizeof(int));
    cudaMalloc(&d_x, N * sizeof(int));
    cudaMalloc(&d_y, N * sizeof(int));

    cudaMemcpy(d_values, h_values, 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colInd, h_colInd, 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rowPtr, h_rowPtr, 4 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, N * sizeof(int), cudaMemcpyHostToDevice);

    spmv_csr<<<1, N>>>(d_values, d_rowPtr, d_colInd, d_x, d_y, N);
    cudaMemcpy(h_y, d_y, N * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) printf("y[%d] = %d\n", i, h_y[i]);

    cudaFree(d_values); cudaFree(d_colInd); cudaFree(d_rowPtr);
    cudaFree(d_x); cudaFree(d_y);
    return 0;
}
