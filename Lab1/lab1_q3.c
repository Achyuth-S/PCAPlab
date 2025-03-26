#include <stdio.h>
#include <mpi.h>

void add(double a, double b) {
    printf("Addition: %f + %f = %f\n", a, b, a + b);}

void subtract(double a, double b) {
    printf("Subtraction: %f - %f = %f\n", a, b, a - b);}

void multiply(double a, double b) {
    printf("Multiplication: %f * %f = %f\n", a, b, a * b);}

void divide(double a, double b) {
    if (b != 0) {
        printf("Division: %f / %f = %f\n", a, b, a / b);
    } else {
        printf("Division by zero error!\n");}
}

int main(int argc, char *argv[]) {
    int rank, size;
    double a, b;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 4) {
        if (rank == 0) {
            printf("Atleast 4 processes.\n");
        }
        MPI_Finalize();
        return 0;
    }

    if (rank == 0) {
        printf("Enter two numbers: ");
        fflush(stdout); 
        scanf("%lf %lf", &a, &b);

        for (int i = 1; i < 4; i++) {
            MPI_Send(&a, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
            MPI_Send(&b, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(&a, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&b, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    switch (rank) {
        case 0:
            add(a, b);
            break;
        case 1:
            subtract(a, b);
            break;
        case 2:
            multiply(a, b);
            break;
        case 3:
            divide(a, b);
            break;
        default:
            printf("Unused process with rank %d\n", rank);
            break;
    }
    MPI_Finalize();
    return 0;
}


/*mpicc -o lab1_q3 lab1_q3.c
mpirun -np 4 ./lab1_q3*/

/*Enter two numbers: 2
7
Addition: 2.000000 + 7.000000 = 9.000000
Subtraction: 2.000000 - 7.000000 = -5.000000
Multiplication: 2.000000 * 7.000000 = 14.000000
Division: 2.000000 / 7.000000 = 0.285714
*/

