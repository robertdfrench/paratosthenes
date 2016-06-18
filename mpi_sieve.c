#include "mpi.h"
#include "stdio.h"
#include "stdlib.h"

#include "mpi_sieve.h"

#define PRIMES_TAG 123
#define COMPOSITE 0

void sieve(int* primes, int num_primes, int* composites, int num_composites) {
	int pi = 0;
	while(pi < num_primes) {
		int ci = 0;
		int p = primes[pi];
		while(ci < num_composites) {
			int c = composites[ci];
			if (c % p == 0) {
				composites[ci] = COMPOSITE;
			}
			ci++;
		}
		pi++;
	}
}

void initialize_range(int* ints, int num_ints, int lower_bound) {
	int i = 0;
	while(i < num_ints) {
		ints[i] = i + lower_bound;
		i++;
	}
}

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

void drop_ints(int* ints, int num_ints) {
	MPI_Send(ints, num_ints, MPI_INT, next_worker_id, PRIMES_TAG, MPI_COMM_WORLD);
}

void catch_ints(int* ints, int num_ints) {
	MPI_Status status;
	MPI_Recv(ints, num_ints, MPI_INT, previous_worker_id, PRIMES_TAG, MPI_COMM_WORLD, &status);
}

void top_main(int rank) {
	int primes[3];
	primes[0] = 2;
	primes[1] = 3;
	primes[2] = 5;
	printf("Hello from the top\n");
	fflush(stdout);

	int* composites = (int*)malloc(sizeof(int) * 100);
	initialize_range(composites, 100, 6);
	sieve(primes, 3, composites, 100);
	drop_ints(composites, 100);
}

int count_nonzeros(int* ints, int num_ints) {
	int num_nonzeros = 0;
	int i = 0;
	while (i < num_ints) {
		if (ints[i] != 0) {
			num_nonzeros++;
		}
		i++;
	}
	return num_nonzeros;
}

void middle_main(int rank) {
	int primes[3];
	int* composites = (int*)malloc(sizeof(int) * 100);
	catch_ints(composites, 100);

	int num_relative_primes = count_nonzeros(composites, 100);
	int* composites_pool = (int*)malloc(sizeof(int) * 100);
	
	
	print_ints(primes, 3);
	drop_ints(primes, 3);
}

void bottom_main(int rank) {
	int primes[3];
	catch_ints(primes, 3);
	printf("You have reached the end\n");
}
