#include <cuda_runtime.h>
#include <stdio.h>

// Kernel for row-wise computation
__global__ void addRows(float *A, float *B, float *C, int n) {
    int row = blockIdx.x; // Each block represents one row
    for (int col = 0; col < n; col++) {
        C[row * n + col] = A[row * n + col] + B[row * n + col];
    }
}

// Kernel for column-wise computation
__global__ void addColumns(float *A, float *B, float *C, int n) {
    int col = blockIdx.x; // Each block represents one column
    for (int row = 0; row < n; row++) {
        C[row * n + col] = A[row * n + col] + B[row * n + col];
    }
}

// Kernel for element-wise computation
__global__ void addElements(float *A, float *B, float *C, int n) {
    int row = blockIdx.x;  // Block represents row
    int col = threadIdx.x; // Thread represents column
    C[row * n + col] = A[row * n + col] + B[row * n + col];
}

int main() {
    int n, choice;

    printf("Enter matrix size (n x n): ");
    scanf("%d", &n);

    size_t size = n * n * sizeof(float);
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    printf("Enter elements of matrix A:\n");
    for (int i = 0; i < n * n; i++) scanf("%f", &h_A[i]);

    printf("Enter elements of matrix B:\n");
    for (int i = 0; i < n * n; i++) scanf("%f", &h_B[i]);

    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    do {
        printf("\n1. Row-wise\n");
        printf("2. Column-wise\n");
        printf("3. Element-wise\n");
        printf("4. Exit\n");
        printf("Enter choice: ");
        scanf("%d", &choice);

        if (choice == 1) {
            printf("\nRow-Wise\n");
            addRows<<<n, 1>>>(d_A, d_B, d_C, n);
            
        } else if (choice == 2) {
            printf("\nColumn-Wise\n");
            addColumns<<<n, 1>>>(d_A, d_B, d_C, n);
        } else if (choice == 3) {
            printf("\nElement-Wise\n");
            addElements<<<n, n>>>(d_A, d_B, d_C, n);
        } else if (choice == 4) {
            printf("Exited\n");
            break;
        } else {
            printf("Invalid\n");
            continue;
        }

        cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

        printf("Resultant Matrix:\n");
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                printf("%.2f ", h_C[i * n + j]);
            }
            printf("\n");
        }
    } while (choice != 4);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}

