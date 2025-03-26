#include <stdio.h>
#include <string.h>

__global__ void reverseWords(char *input, char *output) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Find start and end of the current word.
    if (input[idx] != ' ' && (idx == 0 || input[idx - 1] == ' ')) {
        int start = idx;
        while (input[idx] != ' ' && input[idx] != '\0') idx++;
        int end = idx - 1;

        // Reverse the word.
        for (int i = start; i <= end; i++) {
            output[i] = input[end - (i - start)];
        }
        __syncthreads();
        
        // Copy spaces and null terminator.
        if (input[idx] == ' ') output[idx] = ' ';
        else if (input[idx] == '\0') output[idx] = '\0';
    }
}

int main() {
    char input[] = "Hello CUDA World";
    char output[strlen(input) + 1];

    char *d_input, *d_output;

    cudaMalloc((void **)&d_input, sizeof(input));
    cudaMalloc((void **)&d_output, sizeof(output));

    cudaMemcpy(d_input, input, sizeof(input), cudaMemcpyHostToDevice);

    reverseWords<<<1, strlen(input)>>>(d_input, d_output);

    cudaMemcpy(output, d_output, sizeof(output), cudaMemcpyDeviceToHost);

    printf("Reversed words: %s\n", output);

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
