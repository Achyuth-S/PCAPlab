#include <stdio.h>

#define M 3
#define N 3

__global__ void transformMatrix(int *mat) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    int idx = row * N + col;
    if (row == 0)
        mat[idx] = mat[idx]; // same
    else if (row == 1)
        mat[idx] = mat[idx] * mat[idx]; // square
    else if (row == 2)
        mat[idx] = mat[idx] * mat[idx] * mat[idx]; // cube
}

int main() {
    int h_mat[M*N] = {1,2,3, 4,5,6, 7,8,9};
    int *d_mat;

    cudaMalloc(&d_mat, M*N*sizeof(int));
    cudaMemcpy(d_mat, h_mat, M*N*sizeof(int), cudaMemcpyHostToDevice);

    transformMatrix<<<M, N>>>(d_mat);

    cudaMemcpy(h_mat, d_mat, M*N*sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++)
            printf("%d ", h_mat[i*N + j]);
        printf("\n");
    }

    cudaFree(d_mat);
    return 0;
}
