#include<stdio.h>
#include<mpi.h>
#include<ctype.h>
#include<string.h>

void toggle_case(char *str) {
    for (int i = 0; str[i]; i++) {
        if (islower(str[i])) {
            str[i] = toupper(str[i]);
        } else if (isupper(str[i])) {
            str[i] = tolower(str[i]);
        }}}

int main(int argc, char *argv[]) {
    int rank, size;
    char word[100], modified_word[100];
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size!=2){
        if(rank==0) {
            printf("Error\n");
        }
        MPI_Finalize();
        return 0;
    }

    if (rank == 0) {
        printf("Enter a word: ");
        fflush(stdout);
        scanf("%s", word);

        MPI_Ssend(word, strlen(word) + 1, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
        MPI_Recv(modified_word, 100, MPI_CHAR, 1, 1, MPI_COMM_WORLD, &status);

        printf("Modified word: %s\n", modified_word);
    } else if(rank == 1){
        MPI_Recv(word, 100,MPI_CHAR, 0, 0, MPI_COMM_WORLD, &status);
        toggle_case(word);
        MPI_Ssend(word, strlen(word) + 1, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return 0;
}


/*mpicc -o q1 q1.c
mpirun -np 2 ./q1

Enter a word: GOAT
Modified word: goat

*/
