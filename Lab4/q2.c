#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    int rank, size, matrix[3][3] = {{1, 2, 3}, {4, 1, 6}, {7, 8, 1}}, element, count = 0, total_count;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("Enter element to search: ");
        scanf("%d", &element);
    }
    MPI_Bcast(&element, 1, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = rank; i < 3; i += size)
        for (int j = 0; j < 3; j++)
            if (matrix[i][j] == element) count++;

    MPI_Reduce(&count, &total_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (rank == 0)
        printf("Occurrences of %d: %d\n", element, total_count);
    
    MPI_Finalize();
    return 0;
}