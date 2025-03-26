__global__ void addVectors(float *A, float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

int main() {
    int N = 1024;
    float A[N], B[N], C[N];
    for (int i = 0; i < N; i++) { A[i] = i; B[i] = i * 2; }
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));
    
    cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    addVectors<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    
    cudaMemcpy(C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < 10; i++)  // Print first 10 values
        printf("%f ", C[i]);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}