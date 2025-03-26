#include <stdio.h>
#include <cuda.h>

__global__ void convolution_1D(float *N, float *M, float *P, int Mask_Width, int Width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float Pvalue = 0.0;
    int N_start_point = i - (Mask_Width / 2);

    for (int j = 0; j < Mask_Width; j++) {
        if (N_start_point + j >= 0 && N_start_point + j < Width) {
            Pvalue += N[N_start_point + j] * M[j];
        }
    }
    P[i] = Pvalue;
}

int main() {
    int Width = 5, Mask_Width = 3;
    float N[5] = {1, 2, 3, 4, 5};
    float M[3] = {1, 0, -1};
    float P[5];

    float *d_N, *d_M, *d_P;
    cudaMalloc((void **)&d_N, Width * sizeof(float));
    cudaMalloc((void **)&d_M, Mask_Width * sizeof(float));
    cudaMalloc((void **)&d_P, Width * sizeof(float));

    cudaMemcpy(d_N, N, Width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M, M, Mask_Width * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (Width + blockSize - 1) / blockSize;
    convolution_1D<<<numBlocks, blockSize>>>(d_N, d_M, d_P, Mask_Width, Width);

    cudaMemcpy(P, d_P, Width * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Resultant Array:\n");
    for (int i = 0; i < Width; i++) {
        printf("%f ", P[i]);
    }

    cudaFree(d_N);
    cudaFree(d_M);
    cudaFree(d_P);

    return 0;
}
