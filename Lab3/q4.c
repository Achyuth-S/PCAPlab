#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char* argv[]) {
    int rank, size, len, chunk_size;
    char *S1 = NULL, *S2 = NULL, *sub_S1 = NULL, *sub_S2 = NULL, *sub_result = NULL, *result = NULL;

    MPI_Init(&argc, &argv);               // Initialize MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get process rank
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get number of processes

    if (rank == 0) {
        // Read input strings
        printf("Enter String S1: ");
        char temp1[1000], temp2[1000];
        scanf("%s", temp1);
        printf("Enter String S2: ");
        scanf("%s", temp2);

        // Ensure both strings have the same length
        if (strlen(temp1) != strlen(temp2)) {
            printf("Error: Strings must be of the same length.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Get length and check divisibility by N
        len = strlen(temp1);
        if (len % size != 0) {
            printf("Error: String length must be evenly divisible by %d.\n", size);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Allocate memory and copy strings
        S1 = (char*)malloc(len + 1);
        S2 = (char*)malloc(len + 1);
        strcpy(S1, temp1);
        strcpy(S2, temp2);
    }

    // Broadcast length of the strings to all processes
    MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Compute chunk size
    chunk_size = len / size;

    // Allocate memory for each process's substring
    sub_S1 = (char*)malloc(chunk_size + 1);
    sub_S2 = (char*)malloc(chunk_size + 1);
    sub_result = (char*)malloc(2 * chunk_size + 1); // Resultant substring (double size)

    // Scatter substrings of S1 and S2 to all processes
    MPI_Scatter(S1, chunk_size, MPI_CHAR, sub_S1, chunk_size, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Scatter(S2, chunk_size, MPI_CHAR, sub_S2, chunk_size, MPI_CHAR, 0, MPI_COMM_WORLD);

    sub_S1[chunk_size] = '\0';
    sub_S2[chunk_size] = '\0';

    // Interleave characters from sub_S1 and sub_S2
    for (int i = 0; i < chunk_size; i++) {
        sub_result[2 * i] = sub_S1[i];
        sub_result[2 * i + 1] = sub_S2[i];
    }
    sub_result[2 * chunk_size] = '\0';

    // Allocate memory for gathering results in the root process
    if (rank == 0) {
        result = (char*)malloc(2 * len + 1);
    }

    // Gather interleaved substrings at root process
    MPI_Gather(sub_result, 2 * chunk_size, MPI_CHAR, result, 2 * chunk_size, MPI_CHAR, 0, MPI_COMM_WORLD);

    // Root process prints the resultant string
    if (rank == 0) {
        result[2 * len] = '\0';
        printf("Resultant String: %s\n", result);
        free(S1);
        free(S2);
        free(result);
    }

    free(sub_S1);
    free(sub_S2);
    free(sub_result);
    MPI_Finalize();
    return 0;
}