#include <stdio.h>
#include <cuda.h>

__global__ void oddEvenSort(int *a, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int phase = 0; phase < n; phase++) {
        if ((phase % 2 == 0 && tid % 2 == 0 && tid < n - 1) || 
            (phase % 2 == 1 && tid % 2 == 1 && tid < n - 1)) {
            if (a[tid] > a[tid + 1]) {
                int temp = a[tid];
                a[tid] = a[tid + 1];
                a[tid + 1] = temp;
            }
        }
        __syncthreads();
    }
}

int main() {
    int n = 8;
    int h_a[] = {7, 3, 8, 6, 2, 5, 4, 1};

    int *d_a;
    cudaMalloc((void **)&d_a, n * sizeof(int));
    cudaMemcpy(d_a, h_a, n * sizeof(int), cudaMemcpyHostToDevice);

    oddEvenSort<<<1,n>>>(d_a,n);

    cudaMemcpy(h_a,d_a,n*sizeof(int),cudaMemcpyDeviceToHost);

   printf("Sorted Array:\n");
   for(int i=0;i<n;++i){
      printf("%d ",h_a[i]);
   }

   cudaFree(d_a);
   return(0);
}
