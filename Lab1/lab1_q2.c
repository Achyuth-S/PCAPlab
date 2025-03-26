#include<mpi.h>
#include<stdio.h>
int main(int argc,char *argv[]){
  int rank;
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  if( rank % 2 == 0)
    printf("Hello\n");
  else
    printf("World\n");
  MPI_Finalize();
  return 0;
}

//*/mpicc -o lab1_q2 lab1_q2.c -lm/*
//*/mpirun -np 4 ./lab1_q2/*

/*
Hello
World
Hello
World
*/
