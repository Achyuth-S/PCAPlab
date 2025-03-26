#include<mpi.h>
#include<stdio.h>
#include<math.h>

int main(int argc,char** argv){
  int rank,size;
  const int s=2;
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  int result=(int)pow(s, rank);
  printf("Process %d: Pow(%d,%d)= %d\n",rank,s,rank,result);
  MPI_Finalize();
  return 0;
}

/*mpicc -o lab1_q1 lab1_q1.c -lm/*
/*mpirun -np 4 ./lab1/*

/*
Process 0: Pow(2,0)= 1
Process 1: Pow(2,1)= 2
Process 2: Pow(2,2)= 4
Process 3: Pow(2,3)= 8
*/
