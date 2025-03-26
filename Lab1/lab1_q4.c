#include <stdio.h>
#include <mpi.h>
#include <string.h>
#include <ctype.h>

void toggle_case(char *c) {
    if (islower(*c)) {
        *c = toupper(*c);
    } else if (isupper(*c)) {
        *c = tolower(*c);
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    char input[100], output[100];
    int length;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("Enter a string: ");
        fflush(stdout); 
        fgets(input, 100, stdin);
        input[strcspn(input, "\n")] = '\0'; 
        length = strlen(input);

        for (int i = 1; i < size; i++) {
            MPI_Send(&length, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(input, length + 1, MPI_CHAR, i, 0, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(&length, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(input, length + 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    if (rank < length) {
        toggle_case(&input[rank]);
    }

    MPI_Gather(&input[rank], 1, MPI_CHAR, output, 1, MPI_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        output[length] = '\0'; 
        printf("Modified string: %s\n", output);
    }

    MPI_Finalize();
    return 0;
}


/*mpicc -o lab1_q4 lab1_q4.c
mpirun -np 4 ./lab1_q4
*/

/*Enter a string: GOAT
Modified string: goat
*/
