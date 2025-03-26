#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    int rank, size;
    int matrix[4][4], output[4], recv[16];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("Enter a 4x4 matrix:\n");
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                scanf("%d", &matrix[i][j]);
    }

    MPI_Bcast(matrix, 16, MPI_INT, 0, MPI_COMM_WORLD);

    for (int j = 0; j < 4; j++)
        output[j] = matrix[rank][j] + rank;

    MPI_Gather(output, 4, MPI_INT, recv, 4, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Transformed Matrix:\n");
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++)
                printf("%d ", recv[i * 4 + j]);
            printf("\n");
        }
    }

    MPI_Finalize();
    return 0;
}