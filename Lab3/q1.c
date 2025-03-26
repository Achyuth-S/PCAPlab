#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

// Function to compute factorial of a number
long long factorial(int num) {
    long long fact = 1;
    for (int i = 2; i <= num; i++) {
        fact *= i;
    }
    return fact;
}

int main(int argc, char* argv[]) {
    int rank, size, N;
    int *numbers = NULL, recv_num;
    long long fact, sum = 0;
    long long *factorials = NULL;

    MPI_Init(&argc, &argv);               // Initialize MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get process rank
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get number of processes

    if (rank == 0) {
        printf("Enter %d numbers: ", size);
        numbers = (int*)malloc(size * sizeof(int));
        for (int i = 0; i < size; i++) {
            scanf("%d", &numbers[i]);
        }
    }

    // Scatter one value to each process
    MPI_Scatter(numbers, 1, MPI_INT, &recv_num, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Each process computes the factorial of its number
    fact = factorial(recv_num);

    // Gather results from all processes
    if (rank == 0) {
        factorials = (long long*)malloc(size * sizeof(long long));
    }
    MPI_Gather(&fact, 1, MPI_LONG_LONG, factorials, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    // Root process calculates the sum of factorials
    if (rank == 0) {
        for (int i = 0; i < size; i++) {
            sum += factorials[i];
        }
        printf("Sum of factorials: %lld\n", sum);
        free(numbers);
        free(factorials);
    }

    MPI_Finalize();
    return 0;
}