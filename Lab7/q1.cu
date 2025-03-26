#include <stdio.h>
#include <string.h>

__global__ void countWordOccurrences(char *sentence, char *word, int *count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int wordLen = strlen(word);
    int sentLen = strlen(sentence);

    if (idx <= sentLen - wordLen && strncmp(&sentence[idx], word, wordLen) == 0) {
        atomicAdd(count, 1);
    }
}

int main() {
    char sentence[] = "CUDA is great. CUDA is fast. CUDA is powerful.";
    char word[] = "CUDA";
    int count = 0;

    char *d_sentence, *d_word;
    int *d_count;

    cudaMalloc((void **)&d_sentence, sizeof(sentence));
    cudaMalloc((void **)&d_word, sizeof(word));
    cudaMalloc((void **)&d_count, sizeof(int));

    cudaMemcpy(d_sentence, sentence, sizeof(sentence), cudaMemcpyHostToDevice);
    cudaMemcpy(d_word, word, sizeof(word), cudaMemcpyHostToDevice);
    cudaMemcpy(d_count, &count, sizeof(int), cudaMemcpyHostToDevice);

    countWordOccurrences<<<1, strlen(sentence)>>>(d_sentence, d_word, d_count);

    cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

    printf("The word '%s' appears %d times.\n", word, count);

    cudaFree(d_sentence);
    cudaFree(d_word);
    cudaFree(d_count);

    return 0;
}
