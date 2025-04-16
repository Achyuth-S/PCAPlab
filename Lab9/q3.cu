#include <stdio.h>
#define M 4
#define N 4

__global__ void onesComplementInterior(int *in, int *out) {
    int row = blockIdx.y;
    int col = threadIdx.x;
    int idx = row * N + col;

    if (row == 0 || row == M - 1 || col == 0 || col == N - 1)
        out[idx] = in[idx];  // border stays same
    else
        out[idx] = ~in[idx]; // 1's complement for interior
}

int main() {
    int h_a[M*N] = {
        1, 2, 3, 4,
        6, 5, 8, 3,
        2, 4,10, 1,
        9, 1, 2, 5
    };
    int h_b[M*N];
    int *d_a, *d_b;

    cudaMalloc(&d_a, M*N*sizeof(int));
    cudaMalloc(&d_b, M*N*sizeof(int));

    cudaMemcpy(d_a, h_a, M*N*sizeof(int), cudaMemcpyHostToDevice);
    dim3 grid(1, M);
    onesComplementInterior<<<grid, N>>>(d_a, d_b);

    cudaMemcpy(h_b, d_b, M*N*sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++)
            printf("%3d ", h_b[i*N + j]);
        printf("\n");
    }

    cudaFree(d_a); cudaFree(d_b);
    return 0;
}
