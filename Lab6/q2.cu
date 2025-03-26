#include <stdio.h>
#include <cuda.h>

__device__ void selection_sort(int *data, int left, int right) {
    for (int i = left; i <= right; ++i) {
        int min_val = data[i];
        int min_idx = i;

        for (int j = i + 1; j <= right; ++j) {
            if (data[j] < min_val) {
                min_idx = j;
                min_val = data[j];
            }
        }
        if (i != min_idx) {
            data[min_idx] = data[i];
            data[i] = min_val;
        }
    }
}

__global__ void parallel_selection_sort(int *data, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n / 2) {
        selection_sort(data, tid * 2, tid * 2 + 1);
    }
}

int main() {
    int n = 8;
    int h_data[] = {7, 3, 8, 6, 2, 5, 4, 1};
    
    int *d_data;
    cudaMalloc((void **)&d_data, n * sizeof(int));
    cudaMemcpy(d_data, h_data, n * sizeof(int), cudaMemcpyHostToDevice);

    parallel_selection_sort<<<1, n / 2>>>(d_data, n);

    cudaMemcpy(h_data, d_data, n * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Sorted Array:\n");
    for (int i = 0; i < n; ++i) {
        printf("%d ", h_data[i]);
    }

    cudaFree(d_data);
    
    return 0;
}
