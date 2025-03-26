#include <stdio.h>
#include <string.h>

__global__ void concatenateString(char *Sin, char *Sout, int len, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < len * N) {
        Sout[idx] = Sin[idx % len]; // Copy Sin character at correct position in Sout.
    }
}

int main() {
    char Sin[] = "Hello";
    int N = 3; // Number of times to concatenate.
    int len = strlen(Sin);
    char Sout[len * N + 1]; // Output string size: len * N + null terminator.

    char *d_Sin, *d_Sout;
    cudaMalloc((void **)&d_Sin, sizeof(Sin));
    cudaMalloc((void **)&d_Sout, sizeof(Sout));

    cudaMemcpy(d_Sin, Sin, sizeof(Sin), cudaMemcpyHostToDevice);

    int numThreads = len * N;
    concatenateString<<<1, numThreads>>>(d_Sin, d_Sout, len, N);

    cudaMemcpy(Sout, d_Sout, sizeof(Sout), cudaMemcpyDeviceToHost);
    Sout[len * N] = '\0'; // Null-terminate the output string.

    printf("Output string: %s\n", Sout);

    cudaFree(d_Sin);
    cudaFree(d_Sout);

    return 0;
}
