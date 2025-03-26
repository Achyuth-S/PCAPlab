#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>

// Function to check if a character is a vowel
int is_vowel(char ch) {
    ch = tolower(ch);
    return (ch == 'a' || ch == 'e' || ch == 'i' || ch == 'o' || ch == 'u');
}

int main(int argc, char* argv[]) {
    int rank, size, local_count = 0, total_count = 0;
    char *str = NULL, *sub_str = NULL;
    int len, chunk_size, *counts = NULL;

    MPI_Init(&argc, &argv);               // Initialize MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get process rank
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get number of processes

    if (rank == 0) {
        // Read input string
        printf("Enter a string: ");
        char temp[1000];
        scanf("%s", temp);

        // Get length of string and check divisibility by N
        len = strlen(temp);
        if (len % size != 0) {
            printf("Error: String length must be evenly divisible by %d.\n", size);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Allocate memory and copy string
        str = (char*)malloc(len + 1);
        strcpy(str, temp);
    }

    // Broadcast length of the string to all processes
    MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Calculate chunk size
    chunk_size = len / size;

    // Allocate memory for substring
    sub_str = (char*)malloc(chunk_size + 1);

    // Scatter the string to all processes
    MPI_Scatter(str, chunk_size, MPI_CHAR, sub_str, chunk_size, MPI_CHAR, 0, MPI_COMM_WORLD);
    sub_str[chunk_size] = '\0'; // Null-terminate the substring

    // Count non-vowel characters in received substring
    for (int i = 0; i < chunk_size; i++) {
        if (!is_vowel(sub_str[i])) {
            local_count++;
        }
    }

    // Allocate memory for gathering counts
    if (rank == 0) {
        counts = (int*)malloc(size * sizeof(int));
    }

    // Gather counts from all processes
    MPI_Gather(&local_count, 1, MPI_INT, counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Root process prints counts and total sum
    if (rank == 0) {
        for (int i = 0; i < size; i++) {
            printf("Process %d found %d non-vowel(s).\n", i, counts[i]);
            total_count += counts[i];
        }
        printf("Total non-vowels: %d\n", total_count);
        free(str);
        free(counts);
    }

    free(sub_str);
    MPI_Finalize();
    return 0;
}