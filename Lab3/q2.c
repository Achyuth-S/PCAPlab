#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
    int rank, size, M;
    double local_avg, total_avg;
    int *ID = NULL, *recv_data = NULL;
    double *avg_values = NULL;

    MPI_Init(&argc, &argv);               // Initialize MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get process rank
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get number of processes (N)

    if (rank == 0) {
        // Root process reads M
        printf("Enter M: ");
        scanf("%d", &M);

        // Allocate memory for N*M elements
        ID = (int*)malloc(size * M * sizeof(int));

        // Read ID array elements
        printf("Enter %d elements: ", size * M);
        for (int i = 0; i < size * M; i++) {
            scanf("%d", &ID[i]);
        }
    }

    // Broadcast M to all processes
    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate memory for receiving M elements
    recv_data = (int*)malloc(M * sizeof(int));

    // Scatter M elements to each process
    MPI_Scatter(ID, M, MPI_INT, recv_data, M, MPI_INT, 0, MPI_COMM_WORLD);

    // Each process computes its local average
    double sum = 0;
    for (int i = 0; i < M; i++) {
        sum += recv_data[i];
    }
    local_avg = sum / M;

    // Allocate memory for gathering averages at root
    if (rank == 0) {
        avg_values = (double*)malloc(size * sizeof(double));
    }

    // Gather all local averages at root
    MPI_Gather(&local_avg, 1, MPI_DOUBLE, avg_values, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Root calculates total average
    if (rank == 0) {
        total_avg = 0;
        for (int i = 0; i < size; i++) {
            total_avg += avg_values[i];
        }
        total_avg /= size;
        printf("Total average: %.2f\n", total_avg);
        free(ID);
        free(avg_values);
    }

    free(recv_data);
    MPI_Finalize();
    return 0;
}