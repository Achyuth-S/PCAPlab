#include <stdio.h>
#include <string.h>

__global__ void repeatCharacters(char *Sin, char *T, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < len) {
        for (int i = 0; i <= idx; i++) { // Repeat character idx+1 times.
            T[idx * (idx + 1) / 2 + i] = Sin[idx];
        }
    }
}

int main() {
    char Sin[] = "Hai";
    int len = strlen(Sin);
    
    // Calculate size of output string T based on repetition pattern.
    int T_size = len * (len + 1) / 2 + 1; // Sum of first len natural numbers + null terminator.
    char T[T_size];

    char *d_Sin, *d_T;
    
    cudaMalloc((void **)&d_Sin, sizeof(Sin));
    cudaMalloc((void **)&d_T, T_size);

    cudaMemcpy(d_Sin, Sin, sizeof(Sin), cudaMemcpyHostToDevice);

    repeatCharacters<<<1, len>>>(d_Sin, d_T, len);

    cudaMemcpy(T, d_T, T_size, cudaMemcpyDeviceToHost);
    T[T_size - 1] = '\0'; // Null-terminate the output string.

    printf("Output string: %s\n", T);

    cudaFree(d_Sin);
    cudaFree(d_T);

    return 0;
}
