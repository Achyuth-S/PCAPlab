#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

long long factorial(int n) {
    long long result = 1;
    for (int i = 1; i <= n; ++i) {
        result *= i;
    }
    return result;
}

void fibonacci(int n) {
    long long a = 0, b = 1, c;
    printf("Fibonacci sequence up to rank %d: ", n);
    if (n == 0) {
        printf("%lld\n", a);
        return;
    }
    printf("%lld %lld", a, b);
    for (int i = 2; i <= n; ++i) {
        c = a + b;
        a = b;
        b = c;
        printf(" %lld", c);
    }
    printf("\n");
}
 
int main(int argc, char **argv) {
    int rank, size;
 
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
 
    if (rank % 2 == 0) {
        printf("Process %d (even): Factorial of %d is %lld\n", rank, rank, factorial(rank));
    } else {
        printf("Process %d (odd): ", rank);
        fibonacci(rank);
    }
 
    MPI_Finalize();
    return 0;
}


/* mpicc -o lab1_q5 lab1_q5.c
    mpirun -np 4 ./lab1_q5
*/

/*Process 0 (even): Factorial of 0 is 1
Process 1 (odd): Fibonacci sequence up to rank 1: 0 1
Process 3 (odd): Fibonacci sequence up to rank 3: 0 1 1 2
Process 2 (even): Factorial of 2 is 2
*/
