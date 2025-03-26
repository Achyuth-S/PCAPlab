#include <stdio.h>
#include <cuda_runtime.h>

__global__ void transformString(char *S, char *RS, int len) {
    int idx = threadIdx.x; // Each thread handles a row in RS
    
    if (idx < len) {
        int pos = (idx * (len + 1)); // Position to write in RS (including newline)
        for (int i = 0; i < len - idx; i++) {
            RS[pos + i] = S[i];
        }
        RS[pos + (len - idx)] = '\n'; // Add newline
    }
}

int main() {
    char h_S[] = "PCAP"; // Input string
    int len = sizeof(h_S) - 1; // Length of input string (excluding null terminator)
    int newLen = (len * (len + 1)) / 2 + len; // Sum of decreasing lengths + newlines

    char *d_S, *d_RS;
    char h_RS[newLen + 1]; // Output buffer (+1 for null terminator)

    // Allocate memory on GPU
    cudaMalloc((void **)&d_S, len * sizeof(char));
    cudaMalloc((void **)&d_RS, newLen * sizeof(char));

    // Copy data to GPU
    cudaMemcpy(d_S, h_S, len * sizeof(char), cudaMemcpyHostToDevice);

    // Launch kernel with 'len' threads (one per row)
    transformString<<<1, len>>>(d_S, d_RS, len);

    // Copy back result
    cudaMemcpy(h_RS, d_RS, newLen * sizeof(char), cudaMemcpyDeviceToHost);
    h_RS[newLen] = '\0'; // Null terminate the output string

    printf("Output string RS:\n%s", h_RS);

    // Free memory
    cudaFree(d_S);
    cudaFree(d_RS);

    return 0;
}