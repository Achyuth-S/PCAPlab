#include <mpi.h>
#include <stdio.h>

int factorial(int n) {
    int fact = 1;
    for (int i = 2; i <= n; i++) fact *= i;
    return fact;
}

int main(int argc, char** argv) {
    int rank, size, fact, result;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    fact = factorial(rank + 1);
    MPI_Scan(&fact, &result, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    
    if (rank == size - 1)
        printf("Sum of factorials: %d\n", result);
    
    MPI_Finalize();
    return 0;
}