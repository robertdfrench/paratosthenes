#include "mpi.h"
#include "stdio.h"
#include "stdlib.h"

#include "mpi_sieve.h"

#define PRIMES_TAG 123

void assert_valid_configuration(int comm_size) {
	if (comm_size < 3) {
		printf("Please run with at least 3 processes.\n");
		fflush(stdout);
		MPI_Finalize();
		exit(1);
	}
}

int next_worker_id;
int previous_worker_id;
void setup_topology(int rank) {
	previous_worker_id = rank - 1;
	next_worker_id = rank + 1;
}

int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	assert_valid_configuration(size);
	setup_topology(rank);

	if (rank == 0) {
		top_main(rank);
	} else if (rank < size - 1) {
		middle_main(rank);
	} else {
		bottom_main(rank);
	}
	MPI_Finalize();
	return 0;
}

void print_ints(int* ints, int num_ints) {
	int i = 0;
	while (i < num_ints) {
		printf("%d,", ints[i]);
		i += 1;
	}
	printf("\n");
	fflush(stdout);
}

void top_main(int rank) {
	int primes[3];
	primes[0] = 2;
	primes[1] = 3;
	primes[2] = 5;
	printf("Hello from the top\n");
	fflush(stdout);
	MPI_Send(primes, 3, MPI_INT, next_worker_id, PRIMES_TAG, MPI_COMM_WORLD);
}

void middle_main(int rank) {
	int primes[3];
	MPI_Status status;
	MPI_Recv(primes, 3, MPI_INT, previous_worker_id, PRIMES_TAG, MPI_COMM_WORLD, &status);
	primes[2] += rank;
	printf("Stuck in the middle with you (rank %d): ", rank);
	print_ints(primes, 3);
	MPI_Send(primes, 3, MPI_INT, next_worker_id, PRIMES_TAG, MPI_COMM_WORLD);
}

void bottom_main(int rank) {
	int primes[3];
	MPI_Status status;
	MPI_Recv(primes, 3, MPI_INT, previous_worker_id, PRIMES_TAG, MPI_COMM_WORLD, &status);
	printf("You have reached the end\n");
}
