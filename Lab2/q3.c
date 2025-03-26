#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int *array = NULL;
    int element, result;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int buffer_size = size * (sizeof(int) + MPI_BSEND_OVERHEAD);
    void *buffer = malloc(buffer_size);
    MPI_Buffer_attach(buffer, buffer_size);

    if (rank == 0) {
        array = (int *)malloc(size * sizeof(int));
        printf("Enter %d elements: ", size);
        for (int i = 0; i < size; i++) {
            scanf("%d", &array[i]);
        }

        for (int i = 0; i < size; i++) {
            MPI_Bsend(&array[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }}

    MPI_Recv(&element, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

    if (rank % 2 == 0) {
        result = element * element;
    } else {
        result = element * element * element;
    }

    printf("Process %d received element %d and computed result %d\n", rank, element, result);

    if (rank == 0) {
        free(array);
    }
    MPI_Buffer_detach(&buffer, &buffer_size);
    free(buffer);

    MPI_Finalize();
    return 0;
}

/*
mpicc -o q3 q3.c
mpirun -np 4 ./q3nter 4 elements: 4 3 5 7
Process 0 received element 4 and computed result 16
Process 1 received element 3 and computed result 27
Process 2 received element 5 and computed result 25
Process 3 received element 7 and computed result 343
*/
