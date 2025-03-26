#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size, number;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("Enter a number to send to the slave processes: ");
        fflush(stdout);  
        scanf("%d", &number);
        for (int i = 1; i < size; i++) {
            MPI_Send(&number, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(&number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        printf("Process %d received number %d from master process\n", rank, number);
    }
    MPI_Finalize();
    return 0;
}


/*mpicc -o q2 q2.c
mpirun -np 4 ./q2

Enter a number to send to the slave processes: 55
Process 1 received number 55 from master process
Process 2 received number 55 from master process
Process 3 received number 55 from master process

*/
