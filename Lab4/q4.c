#include <mpi.h>
#include <stdio.h>
#include <string.h>

#define MAX_LEN 100  

void expand_character(char ch, int rank, char *expanded) {
    for (int i = 0; i < rank + 1; i++) {
        expanded[i] = ch;
    }
    expanded[rank + 1] = '\0';  
}

int main(int argc, char *argv[]) {
    int rank, size;
    char input_word[MAX_LEN] = "PCAP";  
    int N = strlen(input_word);
    char expanded[MAX_LEN], final_output[MAX_LEN] = "";

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        for (int i = 1; i < N; i++) {
            MPI_Send(&input_word[i], 1, MPI_CHAR, i, 0, MPI_COMM_WORLD);
        }

        expand_character(input_word[0], 0, expanded);
        strcat(final_output, expanded);

        for (int i = 1; i < N; i++) {
            MPI_Recv(expanded, MAX_LEN, MPI_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            strcat(final_output, expanded);
        }

        printf("Output: %s\n", final_output);

    } else if (rank < N) {
        char received_char;
        MPI_Recv(&received_char, 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        expand_character(received_char, rank, expanded);
        
        MPI_Send(expanded, strlen(expanded) + 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}